# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import secrets
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from .archive import extract_job
from .backend import configured_backend
from .config import NVFLARE_VERSION, Settings, relative_path
from .service import WesService


class LimitBody:
    """Bound the entire multipart request, including chunked uploads, before parsing."""

    def __init__(self, app, limit):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                response = JSONResponse({"msg": "Request body exceeds upload limit", "status_code": 413}, 413)
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings=None, backend=None, background=True):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        service = WesService(settings, backend or configured_backend())
        app.state.service = service
        if background:
            service.start()
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="NVFLARE WES pilot", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(LimitBody, limit=settings.max_upload_bytes)

    def authorize(request: Request):
        value = request.headers.get("authorization", "")
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.encode(), settings.api_token.encode()):
            raise HTTPException(401, "A valid pilot bearer token is required", headers={"WWW-Authenticate": "Bearer"})

    def service(request: Request):
        return request.app.state.service

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            {"msg": str(exc.detail), "status_code": exc.status_code}, exc.status_code, headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({"msg": "Invalid request parameters", "status_code": 400}, 400)

    @app.exception_handler(ValueError)
    async def invalid_value(request, exc):
        return JSONResponse({"msg": "Invalid request or page token", "status_code": 400}, 400)

    @app.exception_handler(KeyError)
    async def missing_run(request, exc):
        return JSONResponse({"msg": "Run or task not found", "status_code": 404}, 404)

    router = APIRouter(prefix="/ga4gh/wes/v1", dependencies=[Depends(authorize)])

    @router.get("/service-info")
    def info(svc=Depends(service)):
        return {
            "id": "org.nvflare.ga4gh.pilot",
            "name": "NVFLARE WES pilot",
            "type": {"group": "org.ga4gh", "artifact": "wes", "version": "1.1.0"},
            "description": "Synthetic-data pilot using an NVFLARE-specific exported job format",
            "organization": {"name": "NVFLARE pilot operator", "url": "https://nvflare.readthedocs.io"},
            "version": "0.1.0",
            "environment": "test",
            "workflow_type_versions": {"NVFLARE": {"workflow_type_version": ["1.0"]}},
            "supported_wes_versions": ["1.1.0"],
            "supported_filesystem_protocols": [],
            "workflow_engine_versions": {"NVFLARE": {"workflow_engine_version": [NVFLARE_VERSION]}},
            "default_workflow_engine_parameters": [],
            "system_state_counts": svc.state_counts(),
            "auth_instructions_url": (
                "https://github.com/NVIDIA/NVFlare/blob/main/integration/ga4gh/"
                "inbound-wes.md#authentication-and-the-governance-roadmap"
            ),
            "tags": {"workflow_profile": "attached-exported-job-zip", "purpose": "synthetic-data-pilot"},
        }

    @router.post("/runs")
    async def submit(request: Request, svc=Depends(service)):
        allowed = {
            "workflow_type",
            "workflow_type_version",
            "workflow_url",
            "workflow_params",
            "tags",
            "workflow_engine",
            "workflow_engine_version",
            "workflow_engine_parameters",
            "workflow_attachment",
        }
        async with request.form(max_files=1, max_fields=10, max_part_size=65536) as form:
            if set(form) - allowed or any(len(form.getlist(key)) != 1 for key in form):
                raise HTTPException(400, "Unknown or repeated form fields")
            if form.get("workflow_type") != "NVFLARE" or form.get("workflow_type_version") != "1.0":
                raise HTTPException(400, "Supported workflow type is NVFLARE version 1.0")
            for field in allowed - {"workflow_attachment"}:
                if field in form and not isinstance(form[field], str):
                    raise HTTPException(400, "Expected text form fields")
            attachment = form.get("workflow_attachment")
            workflow_url = form.get("workflow_url", "")
            relative_path(workflow_url)
            if not isinstance(attachment, UploadFile) or attachment.filename != workflow_url:
                raise HTTPException(400, "workflow_url must reference the single attached job ZIP")
            if not workflow_url.lower().endswith(".zip"):
                raise HTTPException(400, "Expected a ZIP attachment")
            params = json.loads(form.get("workflow_params", "{}"))
            engine_params = json.loads(form.get("workflow_engine_parameters", "{}"))
            tags = json.loads(form.get("tags", "{}"))
            if params != {} or engine_params != {}:
                raise HTTPException(400, "Workflow and engine parameters must be empty objects")
            if not isinstance(tags, dict) or not all(isinstance(v, str) for v in tags.values()):
                raise HTTPException(400, "Tags must be a JSON object with string values")
            engine = form.get("workflow_engine")
            version = form.get("workflow_engine_version")
            if engine not in (None, "NVFLARE") or (version is not None and (not engine or version != NVFLARE_VERSION)):
                raise HTTPException(400, "Unsupported workflow engine or version")
            payload = {
                "workflow_type": "NVFLARE",
                "workflow_type_version": "1.0",
                "workflow_url": workflow_url,
                "workflow_params": {},
                "workflow_engine_parameters": {},
                "tags": tags,
            }
            if engine:
                payload["workflow_engine"] = engine
            if version:
                payload["workflow_engine_version"] = version
            staging = Path(tempfile.mkdtemp(prefix="upload-", dir=svc.root))
            try:
                archive = staging / "job.zip"
                with archive.open("wb") as dst:
                    while chunk := await attachment.read(1024 * 1024):
                        dst.write(chunk)
                job = extract_job(archive, staging / "job", settings)
                archive.unlink()
                run_id = svc.create(payload, staging, job.relative_to(staging))
                return {"run_id": run_id}
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

    @router.get("/runs")
    def list_runs(page_size: int = Query(100, ge=1, le=1000), page_token: str = "", svc=Depends(service)):
        return svc.list_runs(page_size, page_token)

    @router.get("/runs/{run_id}/status")
    def status(run_id: str, svc=Depends(service)):
        return {"run_id": run_id, "state": svc.store.get(run_id)["state"]}

    @router.get("/runs/{run_id}")
    def run_log(run_id: str, svc=Depends(service)):
        return svc.run_log(run_id)

    @router.post("/runs/{run_id}/cancel")
    def cancel(run_id: str, svc=Depends(service)):
        svc.cancel(run_id)
        return {"run_id": run_id}

    @router.get("/runs/{run_id}/tasks")
    def tasks(run_id: str, page_size: int = Query(100, ge=1, le=1000), page_token: str = "", svc=Depends(service)):
        if page_token:
            raise HTTPException(400, "Single-task runs do not have a next page")
        return {"task_logs": [svc.log(svc.store.get(run_id), task=True)], "next_page_token": ""}

    @router.get("/runs/{run_id}/tasks/{task_id}")
    def task(run_id: str, task_id: str, svc=Depends(service)):
        if task_id != "federation":
            raise HTTPException(404, "Task not found")
        return svc.log(svc.store.get(run_id), task=True)

    @router.get("/runs/{run_id}/diagnostics")
    def diagnostics(run_id: str, svc=Depends(service)):
        run = svc.store.get(run_id)
        return {key: run[key] for key in ("run_id", "job_id", "state", "nvflare_status", "diagnostic")}

    @router.get("/runs/{run_id}/outputs/{output_id}")
    def output(run_id: str, output_id: int, svc=Depends(service)):
        run = svc.store.get(run_id)
        if run["state"] != "COMPLETE" or not 0 <= output_id < len(run["output_paths"]):
            raise HTTPException(404, "Output not available")
        return FileResponse(
            svc.root / run_id / "outputs" / str(output_id),
            media_type="application/octet-stream",
            filename=Path(run["output_paths"][output_id]).name,
        )

    app.include_router(router)
    return app

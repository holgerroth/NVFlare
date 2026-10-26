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

"""Real two-client POC, live HTTP WES, optional external TES and an offline visual report."""

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np
from ga4gh_bridge.report import render
from ga4gh_bridge.tes import TERMINAL
from tes_fixture import TesFixture

import nvflare
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.recipe import FedAvgRecipe

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = Path(__file__).resolve().parents[1]
PREFIX = "/ga4gh/wes/v1"
WES_TERMINAL = {"COMPLETE", "CANCELED", "SYSTEM_ERROR", "EXECUTOR_ERROR"}


def wait_for(function, predicate, timeout=240):
    deadline = time.monotonic() + timeout
    value = None
    while time.monotonic() < deadline:
        value = function()
        if predicate(value):
            return value
        time.sleep(0.5)
    raise TimeoutError(f"Timed out; last observed value: {value}")


def recipe(name, client, rounds=3):
    return FedAvgRecipe(
        name=name,
        framework=FrameworkType.NUMPY,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        min_clients=2,
        num_rounds=rounds,
        train_script=str(client),
        launch_external_process=False,
        key_metric="weight_mean",
    )


class LiveWes:
    """Own exactly one Uvicorn process and retain its state across a hard restart."""

    def __init__(self, workdir, kit):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.workdir = workdir
        self.process = None
        self.log = (workdir / "wes.log").open("a")
        token = secrets.token_urlsafe(32)
        self.env = {
            **os.environ,
            "WES_STATE_DIR": str(workdir / "wes"),
            "WES_API_TOKEN": token,
            "WES_PUBLIC_URL": f"http://127.0.0.1:{self.port}{PREFIX}",
            "NVFLARE_ADMIN_USER": "admin@nvidia.com",
            "NVFLARE_STARTUP_KIT": str(kit),
        }
        self.client = httpx.Client(
            base_url=self.env["WES_PUBLIC_URL"] + "/",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )

    def start(self):
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ga4gh_bridge.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                "1",
            ],
            env=self.env,
            cwd=self.workdir,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

        def ready():
            if self.process.poll() is not None:
                raise RuntimeError("WES exited; inspect wes.log")
            try:
                return self.client.get("service-info").status_code == 200
            except httpx.TransportError:
                return False

        wait_for(ready, bool, 60)

    def stop(self, hard=False):
        if self.process and self.process.poll() is None:
            self.process.kill() if hard else self.process.terminate()
            try:
                self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)

    def close(self):
        self.stop()
        self.client.close()
        self.log.close()

    def get(self, path):
        response = self.client.get(path)
        response.raise_for_status()
        return response.json()

    def submit(self, exported, name):
        archive = shutil.make_archive(str(self.workdir / name), "zip", exported, name)
        with open(archive, "rb") as stream:
            response = self.client.post(
                "runs",
                data={"workflow_type": "NVFLARE", "workflow_type_version": "1.0", "workflow_url": "job.zip"},
                files={"workflow_attachment": ("job.zip", stream)},
            )
        response.raise_for_status()
        return response.json()["run_id"]

    def cancel(self, run_id):
        response = self.client.post(f"runs/{run_id}/cancel")
        response.raise_for_status()


def tes_cancel_probe(client, image):
    """Cancel an ordinary sleeping analysis after observing RUNNING at a real TES service."""
    response = client.post(
        "tasks",
        json={
            "name": "ga4gh-validation-cancel",
            "executors": [{"image": image, "command": ["python", "-c", "import time; time.sleep(300)"]}],
            "resources": {"cpu_cores": 1, "ram_gb": 1},
            "tags": {"purpose": "ga4gh-synthetic-cancellation-check"},
        },
    )
    response.raise_for_status()
    task_id = response.json()["id"]

    def status():
        result = client.get(f"tasks/{task_id}", params={"view": "MINIMAL"})
        result.raise_for_status()
        return result.json()["state"]

    try:
        observed = wait_for(status, lambda state: state == "RUNNING" or state in TERMINAL, 120)
        if observed != "RUNNING":
            raise RuntimeError(f"Cancellation probe did not run: {observed}")
    finally:
        response = client.post(f"tasks/{task_id}:cancel")
        response.raise_for_status()
        final = wait_for(status, lambda state: state in TERMINAL, 60)
    if final != "CANCELED":
        raise RuntimeError(f"TES cancellation probe ended in {final}")
    return {"id": task_id, "state": final, "purpose": "running cancellation probe"}


def collect_tasks(client, staging):
    tasks = []
    for path in sorted(staging.glob("*/task.json")):
        record = json.loads(path.read_text())
        task_id = record.get("task_id")
        if not task_id:
            raise RuntimeError("TES submission outcome unresolved; inspect retained task record")
        response = client.get(f"tasks/{task_id}", params={"view": "MINIMAL"})
        response.raise_for_status()
        tasks.append({"id": task_id, "state": response.json()["state"], **record["context"]})
    return tasks


def run(workdir, tes_endpoint=None, label="local"):
    if nvflare.__version__ != "2.9.0" or Path(nvflare.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError("Run with the installed NVFLARE 2.9.0 package, outside the repository root")
    if workdir.exists():
        raise ValueError("Use a new work directory; this test never replaces existing POC workspaces")
    workdir.mkdir(parents=True)
    report_dir = workdir / "report"
    report_dir.mkdir()
    staging = workdir / "tes-staging"
    os.environ.update(
        {
            "NVFLARE_POC_WORKSPACE": str(workdir / "poc"),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "PYTHONPATH": str(INTEGRATION),
            "GA4GH_ANALYSIS_MODE": "tes",
            "GA4GH_TES_SITE": "site-1",
            "TES_STAGING_DIR": str(staging),
            "GA4GH_VALIDATION_TRACE_DIR": str(workdir / "site-traces"),
        }
    )
    provenance = {
        "label": label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nvflare_module": nvflare.__file__,
        "packages": {name: version(name) for name in ("nvflare", "fastapi", "starlette", "httpx", "numpy")},
        "source_sha256": {
            str(path.relative_to(INTEGRATION)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(INTEGRATION.rglob("*.py"))
            if "__pycache__" not in path.parts and ".venv" not in path.parts
        },
    }
    summary = {
        "scope": "end-to-end",
        "rounds": 3,
        "nvflare_version": nvflare.__version__,
        "tes_protocol_fixture": tes_endpoint is None,
        "runs": {},
        "cancellation": {},
        "provenance": provenance,
    }
    exported = workdir / "jobs"
    recipe("hello-numpy", ROOT / "examples/hello-world/hello-numpy/client.py").export(str(exported))
    recipe("ga4gh-numpy", INTEGRATION / "examples/client.py").export(str(exported))
    recipe("cancel-numpy", INTEGRATION / "examples/client.py", rounds=1000).export(str(exported))
    cli = [sys.executable, "-m", "nvflare.cli", "poc"]
    fixture = contextlib.nullcontext() if tes_endpoint else TesFixture(staging)
    with fixture as tes, (workdir / "poc-command.log").open("w") as console:
        os.environ["TES_ENDPOINT"] = tes_endpoint or tes.endpoint
        tes_client = httpx.Client(
            base_url=os.environ["TES_ENDPOINT"].rstrip("/") + "/",
            timeout=10,
            headers={"Authorization": f"Bearer {os.environ['TES_TOKEN']}"} if os.environ.get("TES_TOKEN") else {},
        )
        session = wes = None
        started = False
        try:
            subprocess.run(
                cli + ["prepare", "-n", "2"],
                check=True,
                cwd=workdir,
                stdout=console,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            started = True
            subprocess.run(
                cli + ["start", "--no-wait"],
                check=True,
                cwd=workdir,
                stdout=console,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            kits = list((workdir / "poc").glob("*/prod_00/admin@nvidia.com"))
            assert len(kits) == 1
            session = new_secure_session(
                "admin@nvidia.com", str(kits[0]), timeout=10, command_timeout=30, auto_login_max_tries=1
            )
            wait_for(lambda: session.get_system_info(), lambda info: len(info.client_info) == 2, 60)
            started_at = time.monotonic()
            # Exercise the release's submission-token contract against a real server.
            submission_token = secrets.token_hex(16)
            direct_id = session.submit_job(str(exported / "hello-numpy"), submit_token=submission_token)
            assert session.submit_job(str(exported / "hello-numpy"), submit_token=submission_token) == direct_id
            meta = wait_for(lambda: session.get_job_meta(direct_id), lambda m: m["status"].startswith("FINISHED:"))
            assert meta["status"] == "FINISHED:COMPLETED", meta
            downloaded = Path(session.download_job_result(direct_id, destination=str(workdir / "direct")))
            direct = np.load(downloaded / "workspace/models/server.npy", allow_pickle=False)
            np.save(report_dir / "direct.npy", direct)
            summary["runs"]["direct"] = {
                "job_id": direct_id,
                "model": direct.tolist(),
                "elapsed_seconds": time.monotonic() - started_at,
                "submission_token_reused": True,
            }
            print("Direct hello-numpy and submission-token reuse completed", flush=True)
            wes = LiveWes(workdir, kits[0])
            wes.start()
            assert httpx.get(wes.env["WES_PUBLIC_URL"] + "/runs", timeout=10).status_code == 401
            for key, name in (("wes", "hello-numpy"), ("mixed", "ga4gh-numpy")):
                started_at = time.monotonic()
                rid = wes.submit(exported, name)
                if key == "mixed":
                    before = wait_for(lambda: wes.get(f"runs/{rid}/diagnostics"), lambda r: bool(r["job_id"]))
                    wes.stop(hard=True)
                    wes.start()
                    after = wes.get(f"runs/{rid}/diagnostics")
                    summary["restart_recovered"] = before["job_id"] == after["job_id"]
                log = wait_for(lambda: wes.get(f"runs/{rid}"), lambda r: r["state"] in WES_TERMINAL, 420)
                assert log["state"] == "COMPLETE", log
                response = wes.client.get(f"runs/{rid}/outputs/0")
                response.raise_for_status()
                model = np.load(io.BytesIO(response.content), allow_pickle=False)
                np.save(report_dir / f"{key}.npy", model)
                diagnostic = wes.get(f"runs/{rid}/diagnostics")
                summary["runs"][key] = {
                    "run_id": rid,
                    "job_id": diagnostic["job_id"],
                    "state": log["state"],
                    "model": model.tolist(),
                    "elapsed_seconds": time.monotonic() - started_at,
                }
                print(f"Live HTTP WES {name} completed", flush=True)
            mixed_id = summary["runs"]["mixed"]["job_id"]
            summary["site_rounds"] = [
                trace
                for path in sorted((workdir / "site-traces").glob("*.json"))
                if (trace := json.loads(path.read_text()))["job_id"] == mixed_id
            ]
            for stage in ("early", "running"):
                rid = wes.submit(exported, "cancel-numpy")
                if stage == "running":
                    wait_for(lambda: wes.get(f"runs/{rid}/status"), lambda r: r["state"] == "RUNNING")
                    job_id = wes.get(f"runs/{rid}/diagnostics")["job_id"]
                    # Ensure a remote TES task actually exists before aborting the federation.
                    wait_for(
                        lambda: collect_tasks(tes_client, staging),
                        lambda tasks: any(t["job_id"] == job_id for t in tasks),
                    )
                wes.cancel(rid)
                log = wait_for(lambda: wes.get(f"runs/{rid}/status"), lambda r: r["state"] in WES_TERMINAL)
                summary["cancellation"][stage] = log["state"]
                print(f"WES {stage} cancellation: {log['state']}", flush=True)
            summary["tes_tasks"] = wait_for(
                lambda: collect_tasks(tes_client, staging),
                lambda tasks: bool(tasks) and all(t["state"] in TERMINAL for t in tasks),
                60,
            )
            if tes_endpoint:
                probe = tes_cancel_probe(tes_client, os.environ.get("TES_IMAGE", "python:3.12-slim"))
                summary["tes_cancel_probe"] = probe["state"]
                summary["tes_tasks"].append(probe)
        except BaseException as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if wes:
                wes.close()
            if session:
                session.close()
            if started:
                subprocess.run(cli + ["stop"], cwd=workdir, stdout=console, stderr=subprocess.STDOUT, timeout=60)
            tes_client.close()
            summary["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (workdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
            result = render(summary, report_dir)
    if not result["passed"]:
        raise RuntimeError(f"Result validation failed: {report_dir / 'validation.json'}")
    print(f"PASS: {report_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--tes-endpoint", help="Existing TES base URL; defaults to the local protocol fixture")
    parser.add_argument("--label", default="local", help="Environment label recorded in the report")
    args = parser.parse_args()
    run(args.workdir.resolve(), args.tes_endpoint, args.label)

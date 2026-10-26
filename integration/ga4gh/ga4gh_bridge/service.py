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

import base64
import fcntl
import hashlib
import hmac
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .backend import PermanentBackendError
from .config import Settings
from .store import Store

logger = logging.getLogger(__name__)
TERMINAL = {"COMPLETE", "CANCELED", "EXECUTOR_ERROR", "SYSTEM_ERROR"}
STATES = {
    "SUBMITTED": "QUEUED",
    "APPROVED": "QUEUED",
    "DISPATCHED": "INITIALIZING",
    "RUNNING": "RUNNING",
    "FINISHED:EXECUTION_EXCEPTION": "EXECUTOR_ERROR",
    "FINISHED:ABNORMAL": "SYSTEM_ERROR",
    "FINISHED:CAN_NOT_SCHEDULE": "SYSTEM_ERROR",
    "FINISHED:FAILED_TO_RUN": "SYSTEM_ERROR",
    "FINISHED:ABANDONED": "SYSTEM_ERROR",
}


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WesService:
    def __init__(self, settings: Settings, backend):
        self.settings = settings
        self.backend = backend
        self.root = settings.state_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.owner = (self.root / "owner.lock").open("a")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("State directory already has an owner; use exactly one server worker") from None
        self.store = Store(self.root / "runs.sqlite")
        self.tick_lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._work, name="wes-reconciler", daemon=True)
        self.thread.start()

    def _work(self):
        while not self.stop.is_set():
            self.tick()
            self.stop.wait(self.settings.poll_interval)

    def close(self):
        self.stop.set()
        if self.thread:
            # Never release the owner lock while an old reconciler can still submit work.
            self.thread.join()
        self.backend.close()
        self.store.close()
        self.owner.close()

    def create(self, request, staging, job_relative):
        run_id = uuid.uuid4().hex
        root = self.root / run_id
        os.replace(staging, root)
        run = {
            "run_id": run_id,
            "request": request,
            "job_dir": str(root / job_relative),
            "job_id": None,
            "state": "QUEUED",
            "created_at": now(),
            "start_time": None,
            "end_time": None,
            "submission_started": False,
            "cancel_requested": False,
            "result_failures": 0,
            "diagnostic": "Accepted for submission",
            "nvflare_status": "",
            "output_paths": list(self.settings.output_paths),
        }
        self.store.insert(run)
        return run_id

    def cancel(self, run_id):
        with self.store.lock:
            run = self.store.get(run_id)
            if run["state"] not in TERMINAL:
                self.store.patch(run_id, cancel_requested=True, state="CANCELING")

    def tick(self):
        with self.tick_lock:
            for run in self.store.all():
                if run["state"] in TERMINAL:
                    continue
                try:
                    self._reconcile(run["run_id"])
                except PermanentBackendError:
                    self._finish(run["run_id"], "SYSTEM_ERROR", "NVFLARE rejected the operation; consult operator logs")
                except Exception as exc:
                    # Never expose raw backend exceptions, commands, paths or tokens through WES.
                    logger.warning("Run %s reconciliation failed (%s)", run["run_id"], type(exc).__name__)
                    self.store.patch(
                        run["run_id"], diagnostic="NVFLARE temporarily unavailable; reconciliation will retry"
                    )

    def _finish(self, run_id, state, diagnostic):
        self.store.patch(run_id, state=state, diagnostic=diagnostic, end_time=now())

    def _reconcile(self, run_id):
        run = self.store.get(run_id)
        if not run["job_id"]:
            # A previously attempted submission might have succeeded before the connection failed.
            # Resolve it with the SAME token, even when cancellation has since been requested.
            if run["cancel_requested"] and not run["submission_started"]:
                self._finish(run_id, "CANCELED", "Canceled before NVFLARE submission")
                return
            self.store.patch(run_id, submission_started=True)
            job_id = self.backend.submit(run["job_dir"], run_id)
            run = self.store.patch(run_id, job_id=job_id)
        meta = self.backend.meta(run["job_id"])
        status = meta.get("status", "")
        known = set(STATES) | {"FINISHED:COMPLETED", "FINISHED:ABORTED"}
        self.store.patch(run_id, nvflare_status=status if status in known else "UNKNOWN")
        run = self.store.get(run_id)
        if run["cancel_requested"] and not status.startswith("FINISHED:"):
            self.backend.cancel(run["job_id"])
            self.store.patch(run_id, state="CANCELING", diagnostic="Cancellation requested from NVFLARE")
            return
        if status == "FINISHED:ABORTED":
            state = "CANCELED" if run["cancel_requested"] else "EXECUTOR_ERROR"
            self._finish(run_id, state, "NVFLARE job aborted")
            return
        if status == "FINISHED:COMPLETED":
            try:
                self._publish(run)
            except Exception as exc:
                attempts = run["result_failures"] + 1
                logger.warning("Run %s output collection failed (%s)", run_id, type(exc).__name__)
                self.store.patch(run_id, result_failures=attempts, diagnostic="Awaiting allowlisted outputs")
                if attempts >= self.settings.result_attempts:
                    self._finish(run_id, "SYSTEM_ERROR", "Unable to publish required outputs")
                return
            self._finish(run_id, "COMPLETE", "NVFLARE completed and allowlisted outputs are available")
            return
        state = STATES.get(status, "UNKNOWN")
        if state in TERMINAL:
            self._finish(run_id, state, f"NVFLARE status: {status}")
        else:
            values = {"state": state, "diagnostic": f"NVFLARE status: {state}"}
            if state in ("INITIALIZING", "RUNNING") and not run["start_time"]:
                values["start_time"] = now()
            # Do not erase a cancellation request received concurrently with polling.
            with self.store.lock:
                if self.store.get(run_id)["cancel_requested"]:
                    values["state"] = "CANCELING"
                self.store.patch(run_id, **values)

    def _publish(self, run):
        root = self.root / run["run_id"]
        published = root / "outputs"
        if published.is_dir():
            return  # Atomic publication succeeded before a process restart.
        with tempfile.TemporaryDirectory(prefix="results-", dir=root) as scratch:
            scratch = Path(scratch)
            download_dir = scratch / "download"
            download_dir.mkdir()
            downloaded = self.backend.download(run["job_id"], download_dir).resolve()
            if not downloaded.is_relative_to(download_dir.resolve()):
                raise ValueError("Backend returned a path outside its download directory")
            outputs = scratch / "outputs"
            outputs.mkdir()
            total = 0
            for index, name in enumerate(run["output_paths"]):
                source = downloaded / name
                if any(p.is_symlink() for p in (source, *source.parents)):
                    raise ValueError("Symlink in output path")
                if not source.is_file() or not source.resolve().is_relative_to(downloaded):
                    raise ValueError("Required output missing")
                total += source.stat().st_size
                if total > self.settings.max_output_bytes:
                    raise ValueError("Output size limit exceeded")
                shutil.copyfile(source, outputs / str(index))
            os.replace(outputs, published)

    def log(self, run, task=False):
        result = {
            "name": "NVFLARE federated analysis",
            "system_logs": [run["diagnostic"]],
            "stdout": f"{self.settings.public_url}/runs/{run['run_id']}/diagnostics",
        }
        for key in ("start_time", "end_time"):
            if run[key]:
                result[key] = run[key]
        if run["state"] in TERMINAL:
            result["exit_code"] = {"COMPLETE": 0, "CANCELED": 130}.get(run["state"], 1)
        if task:
            result["id"] = "federation"
        return result

    def run_log(self, run_id):
        run = self.store.get(run_id)
        outputs = {}
        if run["state"] == "COMPLETE":
            outputs = {
                name: f"{self.settings.public_url}/runs/{run_id}/outputs/{index}"
                for index, name in enumerate(run["output_paths"])
            }
        return {
            "run_id": run_id,
            "request": run["request"],
            "state": run["state"],
            "run_log": self.log(run),
            "task_logs_url": f"{self.settings.public_url}/runs/{run_id}/tasks",
            "outputs": outputs,
        }

    def list_runs(self, size, token):
        if token:
            try:
                encoded, signature = token.split(".")
                expected = hmac.new(self.settings.api_token.encode(), encoded.encode(), hashlib.sha256).hexdigest()
                if not hmac.compare_digest(signature, expected):
                    raise ValueError()
                key, offset = json.loads(base64.urlsafe_b64decode(encoded))
                if type(offset) is not int or offset < 0:
                    raise ValueError()
                summaries = self.store.get_snapshot(key)
            except Exception as exc:
                raise ValueError("Invalid or expired page token") from exc
        else:
            summaries = []
            for run in self.store.all():
                summary = {"run_id": run["run_id"], "state": run["state"], "tags": run["request"]["tags"]}
                summary.update({k: run[k] for k in ("start_time", "end_time") if run[k]})
                summaries.append(summary)
            key = self.store.snapshot(summaries)
            offset = 0
        next_token = ""
        if offset + size < len(summaries):
            encoded = base64.urlsafe_b64encode(json.dumps([key, offset + size]).encode()).decode()
            signature = hmac.new(self.settings.api_token.encode(), encoded.encode(), hashlib.sha256).hexdigest()
            next_token = f"{encoded}.{signature}"
        return {"runs": summaries[offset : offset + size], "next_page_token": next_token}

    def state_counts(self):
        return dict(Counter(run["state"] for run in self.store.all()))

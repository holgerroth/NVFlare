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

"""Small TES client for a site-controlled file:// staging area, not a worker launcher."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import analysis

TERMINAL = {"COMPLETE", "EXECUTOR_ERROR", "SYSTEM_ERROR", "CANCELED", "PREEMPTED"}
TASK_ID = re.compile(r"[a-zA-Z0-9_.-]{1,128}\Z")


class TesError(RuntimeError):
    pass


class AmbiguousSubmission(TesError):
    pass


class Cancelled(TesError):
    pass


@dataclass(frozen=True)
class TesSettings:
    endpoint: str
    staging_dir: Path
    token: str = ""
    image: str = "python:3.12-slim"
    request_timeout: float = 10.0
    poll_interval: float = 1.0
    queue_timeout: float = 300.0
    execution_timeout: float = 600.0
    cancel_timeout: float = 30.0
    max_file_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("TES_ENDPOINT must be an HTTP(S) service base URL")
        if any(
            x <= 0
            for x in (
                self.request_timeout,
                self.poll_interval,
                self.queue_timeout,
                self.execution_timeout,
                self.cancel_timeout,
                self.max_file_bytes,
            )
        ):
            raise ValueError("TES timeouts and limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            endpoint=os.environ["TES_ENDPOINT"].rstrip("/"),
            staging_dir=Path(os.environ["TES_STAGING_DIR"]).resolve(),
            token=os.environ.get("TES_TOKEN", ""),
            image=os.environ.get("TES_IMAGE", "python:3.12-slim"),
            queue_timeout=float(os.environ.get("TES_QUEUE_TIMEOUT", "300")),
            execution_timeout=float(os.environ.get("TES_EXECUTION_TIMEOUT", "600")),
        )


class TesRunner:
    def __init__(self, settings: TesSettings, client=None):
        self.settings = settings
        self.root = settings.staging_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.client = client or httpx.Client(
            base_url=settings.endpoint.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {settings.token}"} if settings.token else {},
            timeout=settings.request_timeout,
            follow_redirects=False,
        )

    def close(self):
        self.client.close()

    def _request(self, method, path, **kwargs):
        try:
            response = self.client.request(method, path, timeout=self.settings.request_timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Don't propagate URLs, headers or a remote response body into federation logs.
            raise TesError(f"TES {method} request failed ({type(exc).__name__})") from None

    @staticmethod
    def _save(path, record):
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="record-", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def _task_id(value):
        if not isinstance(value, str) or not TASK_ID.fullmatch(value) or value in (".", ".."):
            raise TesError("TES returned an invalid task ID")
        return value

    def _recover(self, record, record_path):
        if record.get("task_id"):
            return self._task_id(record["task_id"])
        result = self._request(
            "GET", "tasks", params={"tag_key": "nvflare_operation", "tag_value": record["operation"], "view": "MINIMAL"}
        )
        tasks = result.get("tasks", [])
        if len(tasks) != 1 or result.get("next_page_token"):
            raise AmbiguousSubmission(
                "TES submission outcome unresolved; inspect the persisted operation tag before retrying or canceling"
            )
        task_id = self._task_id(tasks[0].get("id"))
        record.update(task_id=task_id, state="RECOVERED")
        self._save(record_path, record)
        return task_id

    def cancel_record(self, record_path):
        """Reconcile and cancel a recorded task, including after loss of the original client process."""
        record_path = Path(record_path).resolve()
        if not record_path.is_relative_to(self.root):
            raise ValueError("Task record must be inside TES_STAGING_DIR")
        record = json.loads(record_path.read_text())
        self._check_endpoint(record)
        task_id = self._recover(record, record_path)
        deadline = time.monotonic() + self.settings.cancel_timeout
        while time.monotonic() < deadline:
            try:
                state = self._request("GET", f"tasks/{task_id}", params={"view": "MINIMAL"}).get("state")
                if state in TERMINAL:
                    record["state"] = state
                    self._save(record_path, record)
                    return
                self._request("POST", f"tasks/{task_id}:cancel")
            except TesError:
                pass
            time.sleep(min(self.settings.poll_interval, max(0, deadline - time.monotonic())))
        record["state"] = "CANCEL_UNCONFIRMED"
        self._save(record_path, record)
        raise TesError("TES cancellation unconfirmed; task record retained for operator cleanup")

    def _check_endpoint(self, record):
        if record["endpoint"] != self.settings.endpoint.rstrip("/"):
            raise TesError("Recorded task belongs to another TES endpoint")

    def run(self, document, context, check_cancel=lambda: None):
        """Run one calculation; context must uniquely identify job, site, task and round."""
        operation = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        with (self.root / f"{operation}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise TesError("Another caller owns this TES operation") from None
            return self._run(document, context, check_cancel)

    def _run(self, document, context, check_cancel):
        if set(context) != {"job_id", "site", "task", "round"} or not all(isinstance(v, str) for v in context.values()):
            raise ValueError("Expected string job_id, site, task and round context")
        payload = json.dumps(document, allow_nan=False, sort_keys=True)
        if len(payload.encode()) > self.settings.max_file_bytes:
            raise ValueError("TES input exceeds size limit")
        operation = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        root = self.root / operation
        root.mkdir(exist_ok=True, mode=0o700)
        record_path = root / "task.json"
        input_path, output_path = root / "input.json", root / "output.json"
        fingerprint = hashlib.sha256((payload + self.settings.image).encode()).hexdigest()
        try:
            check_cancel()
        except Exception:
            if record_path.exists():
                self.cancel_record(record_path)
            raise
        if record_path.exists():
            record = json.loads(record_path.read_text())
            self._check_endpoint(record)
            if record["fingerprint"] != fingerprint:
                raise TesError("Existing operation has different inputs or image")
            task_id = self._recover(record, record_path)
        else:
            input_path.write_text(payload)
            script_path = root / "analysis.py"
            shutil.copyfile(analysis.__file__, script_path)
            task = {
                "name": f"nvflare-{operation[:16]}",
                "tags": {**context, "nvflare_operation": operation},
                "inputs": [
                    {"url": path.as_uri(), "path": f"/work/{path.name}", "type": "FILE"}
                    for path in (input_path, script_path)
                ],
                "outputs": [{"url": output_path.as_uri(), "path": "/work/output.json", "type": "FILE"}],
                "executors": [
                    {
                        "image": self.settings.image,
                        "command": ["python", "/work/analysis.py", "/work/input.json", "/work/output.json"],
                    }
                ],
                "resources": {"cpu_cores": 1, "ram_gb": 1},
            }
            record = {
                "operation": operation,
                "context": context,
                "fingerprint": fingerprint,
                "endpoint": self.settings.endpoint.rstrip("/"),
                "task_id": None,
                "state": "SUBMITTING",
                "created_at": time.time(),
                "running_at": None,
            }
            self._save(record_path, record)  # Persist BEFORE sending the non-idempotent POST.
            try:
                task_id = self._task_id(self._request("POST", "tasks", json=task).get("id"))
            except TesError:
                record["state"] = "SUBMISSION_UNKNOWN"
                self._save(record_path, record)
                raise AmbiguousSubmission(
                    "TES submission outcome unknown; task record retained for reconciliation"
                ) from None
            record.update(task_id=task_id, state="QUEUED")
            self._save(record_path, record)
        terminal = False
        try:
            while True:
                check_cancel()
                started = record["running_at"]
                limit = self.settings.execution_timeout if started else self.settings.queue_timeout
                if record["state"] not in TERMINAL and time.time() - (started or record["created_at"]) > limit:
                    raise TesError("TES execution deadline exceeded" if started else "TES queue deadline exceeded")
                result = self._request("GET", f"tasks/{task_id}", params={"view": "MINIMAL"})
                state = result.get("state", "UNKNOWN")
                if state not in TERMINAL | {"UNKNOWN", "QUEUED", "INITIALIZING", "RUNNING", "PAUSED", "CANCELING"}:
                    raise TesError("TES returned an unsupported task state")
                record["state"] = state
                if state == "RUNNING" and not record["running_at"]:
                    record["running_at"] = time.time()
                self._save(record_path, record)
                if state in TERMINAL:
                    terminal = True
                    if state != "COMPLETE":
                        raise TesError(f"TES analysis ended in {state}")
                    check_cancel()
                    if output_path.is_symlink() or not output_path.is_file():
                        raise TesError("TES output is missing or is a symlink")
                    if output_path.stat().st_size > self.settings.max_file_bytes:
                        raise TesError("TES output exceeds size limit")
                    return json.loads(output_path.read_text())
                time.sleep(self.settings.poll_interval)
        finally:
            if not terminal:
                self.cancel_record(record_path)


def main():
    parser = argparse.ArgumentParser(description="Cancel/reconcile a TES task retained after an interrupted client")
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    runner = TesRunner(TesSettings.from_env())
    try:
        runner.cancel_record(args.record)
    finally:
        runner.close()


if __name__ == "__main__":
    main()

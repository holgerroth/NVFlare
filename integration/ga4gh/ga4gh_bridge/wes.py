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

"""Consume an existing institution's WES service from an NVFLARE Client API script."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

TERMINAL = {"COMPLETE", "CANCELED", "EXECUTOR_ERROR", "SYSTEM_ERROR"}
STATES = TERMINAL | {"UNKNOWN", "QUEUED", "INITIALIZING", "RUNNING", "PAUSED", "CANCELING"}
RUN_ID = re.compile(r"[a-zA-Z0-9_-]{1,128}\Z")


class WesError(RuntimeError):
    pass


class AmbiguousSubmission(WesError):
    pass


@dataclass(frozen=True)
class WesSettings:
    endpoint: str
    state_dir: Path
    workflow: Path
    token: str = ""
    output_dir: Path | None = None
    workflow_type: str = "CWL"
    workflow_version: str = "v1.2"
    request_timeout: float = 10.0
    poll_interval: float = 1.0
    queue_timeout: float = 300.0
    execution_timeout: float = 600.0
    cancel_timeout: float = 30.0
    max_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        parsed = urlsplit(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query:
            raise ValueError("SITE_WES_ENDPOINT must be an HTTP(S) service base URL without credentials or query")
        if not self.workflow.is_file():
            raise ValueError("SITE_WES_WORKFLOW must name a local workflow file")
        if any(
            value <= 0
            for value in (
                self.request_timeout,
                self.poll_interval,
                self.queue_timeout,
                self.execution_timeout,
                self.cancel_timeout,
                self.max_bytes,
            )
        ):
            raise ValueError("WES timeouts and limits must be positive")

    @classmethod
    def from_env(cls):
        output = os.environ.get("SITE_WES_OUTPUT_DIR")
        return cls(
            endpoint=os.environ["SITE_WES_ENDPOINT"].rstrip("/"),
            state_dir=Path(os.environ["SITE_WES_STATE_DIR"]),
            workflow=Path(os.environ["SITE_WES_WORKFLOW"]),
            token=os.environ.get("SITE_WES_TOKEN", ""),
            output_dir=Path(output) if output else None,
            queue_timeout=float(os.environ.get("SITE_WES_QUEUE_TIMEOUT", "300")),
            execution_timeout=float(os.environ.get("SITE_WES_EXECUTION_TIMEOUT", "600")),
        )


class WesRunner:
    def __init__(self, settings: WesSettings, client=None):
        self.settings = settings
        self.root = settings.state_dir.resolve()
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
            with self.client.stream(method, path, timeout=self.settings.request_timeout, **kwargs) as response:
                response.raise_for_status()
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > self.settings.max_bytes:
                        raise WesError("WES response exceeds size limit")
                result = json.loads(body)
                if not isinstance(result, dict):
                    raise WesError("Expected a JSON object from WES")
                return result
        except (httpx.HTTPError, ValueError) as exc:
            raise WesError(f"WES {method} request failed ({type(exc).__name__})") from None

    @staticmethod
    def _save(path, record):
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _run_id(value):
        if not isinstance(value, str) or not RUN_ID.fullmatch(value):
            raise WesError("WES returned an invalid run ID")
        return value

    def _record(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise WesError("Run record must be inside SITE_WES_STATE_DIR")
        record = json.loads(path.read_text())
        if record["endpoint"] != self.settings.endpoint.rstrip("/"):
            raise WesError("Recorded run belongs to another WES endpoint")
        return record

    def adopt(self, record_path, run_id):
        """Bind an operator-discovered run after an ambiguous POST, verifying its correlation tags."""
        record_path = Path(record_path)
        record = self._record(record_path)
        run_id = self._run_id(run_id)
        if record.get("run_id") not in (None, run_id):
            raise WesError("Record already belongs to a different run")
        log = self._request("GET", f"runs/{run_id}")
        tags = log.get("request", {}).get("tags", {})
        if any(tags.get(key) != record[key] for key in ("nvflare_operation", "nvflare_request_sha256")):
            raise WesError("Run correlation tags do not match the retained record")
        record.update(run_id=run_id)
        self._save(record_path, record)

    def cancel_record(self, record_path):
        record_path = Path(record_path)
        record = self._record(record_path)
        run_id = self._known_run(record)
        deadline = time.monotonic() + self.settings.cancel_timeout
        while time.monotonic() < deadline:
            try:
                state = self._request("GET", f"runs/{run_id}/status").get("state")
                if state in TERMINAL:
                    record["state"] = state
                    self._save(record_path, record)
                    return
                self._request("POST", f"runs/{run_id}/cancel")
            except WesError:
                pass
            time.sleep(min(self.settings.poll_interval, max(0, deadline - time.monotonic())))
        record["state"] = "CANCEL_UNCONFIRMED"
        self._save(record_path, record)
        raise WesError("WES cancellation unconfirmed; retained record requires site operator cleanup")

    def _known_run(self, record):
        if not record.get("run_id"):
            raise AmbiguousSubmission(
                "WES submission outcome unknown; locate the tagged run and adopt it before cleanup"
            )
        return self._run_id(record["run_id"])

    def run(self, parameters, context, attachments=None, check_cancel=lambda: None):
        """Submit one site workflow; a repeated context reuses its recorded run, never blindly resubmits."""
        if set(context) != {"job_id", "site", "task", "round"} or not all(isinstance(v, str) for v in context.values()):
            raise ValueError("Expected string job_id, site, task and round context")
        operation = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        with (self.root / f"{operation}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise WesError("Another caller owns this WES operation") from None
            return self._run(parameters, context, operation, attachments or {}, check_cancel)

    def _run(self, parameters, context, operation, attachments, check_cancel):
        payload = json.dumps(parameters, allow_nan=False, sort_keys=True)
        files = {"workflow.cwl": self.settings.workflow, **attachments}
        if "workflow.cwl" in attachments:
            raise ValueError("Attachment name workflow.cwl is reserved")
        contents = {}
        total = len(payload.encode())
        for name, path in files.items():
            if not re.fullmatch(r"[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+", name):
                raise ValueError("Attachments must have simple filenames")
            with Path(path).open("rb") as stream:
                data = stream.read(self.settings.max_bytes + 1)
            total += len(data)
            if total > self.settings.max_bytes:
                raise WesError("WES submission exceeds size limit")
            contents[name] = data
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    payload,
                    self.settings.workflow_type,
                    self.settings.workflow_version,
                    {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())},
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        root = self.root / operation
        root.mkdir(exist_ok=True, mode=0o700)
        record_path = root / "run.json"
        try:
            check_cancel()
        except Exception:
            if record_path.exists():
                self.cancel_record(record_path)
            raise
        if record_path.exists():
            record = self._record(record_path)
            if record["nvflare_request_sha256"] != fingerprint:
                raise WesError("Existing operation has different inputs or workflow code")
            run_id = self._known_run(record)
        else:
            record = {
                "nvflare_operation": operation,
                "nvflare_request_sha256": fingerprint,
                "context": context,
                "endpoint": self.settings.endpoint.rstrip("/"),
                "run_id": None,
                "state": "SUBMITTING",
                "created_at": time.time(),
                "running_at": None,
            }
            self._save(record_path, record)
            try:
                result = self._request(
                    "POST",
                    "runs",
                    data={
                        "workflow_type": self.settings.workflow_type,
                        "workflow_type_version": self.settings.workflow_version,
                        "workflow_url": "workflow.cwl",
                        "workflow_params": payload,
                        "tags": json.dumps(
                            {**context, "nvflare_operation": operation, "nvflare_request_sha256": fingerprint}
                        ),
                    },
                    files=[("workflow_attachment", (name, data)) for name, data in contents.items()],
                )
                run_id = self._run_id(result.get("run_id"))
            except WesError:
                record["state"] = "SUBMISSION_UNKNOWN"
                self._save(record_path, record)
                raise AmbiguousSubmission("WES submission outcome unknown; no automatic resubmission") from None
            record.update(run_id=run_id, state="QUEUED")
            self._save(record_path, record)
        terminal = False
        try:
            while True:
                check_cancel()
                started = record["running_at"]
                limit = self.settings.execution_timeout if started else self.settings.queue_timeout
                if record["state"] not in TERMINAL and time.time() - (started or record["created_at"]) > limit:
                    raise WesError("WES execution deadline exceeded" if started else "WES queue deadline exceeded")
                state = self._request("GET", f"runs/{run_id}/status").get("state")
                if state not in STATES:
                    raise WesError("WES returned an unsupported run state")
                record["state"] = state
                if state == "RUNNING" and not record["running_at"]:
                    record["running_at"] = time.time()
                self._save(record_path, record)
                if state in TERMINAL:
                    terminal = True
                    if state != "COMPLETE":
                        raise WesError(f"WES workflow ended in {state}")
                    check_cancel()
                    log = self._request("GET", f"runs/{run_id}")
                    if log.get("state") != "COMPLETE" or not isinstance(log.get("outputs"), dict):
                        raise WesError("WES completed without a valid output map")
                    return log["outputs"]
                time.sleep(self.settings.poll_interval)
        finally:
            if not terminal:
                self.cancel_record(record_path)

    def read_json_output(self, output):
        """Read a declared JSON result from approved shared storage or the WES HTTP origin."""
        location = output.get("location") if isinstance(output, dict) else output
        if not isinstance(location, str):
            raise WesError("Expected an output URL or CWL File location")
        url = urlsplit(location)
        if url.scheme == "file" and url.netloc in ("", "localhost") and self.settings.output_dir:
            path = Path(unquote(url.path))
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise WesError("Symlink in WES output path")
            if not path.resolve().is_relative_to(self.settings.output_dir.resolve()) or not path.is_file():
                raise WesError("WES output is outside the approved directory or missing")
            with path.open("rb") as stream:
                body = stream.read(self.settings.max_bytes + 1)
            if len(body) > self.settings.max_bytes:
                raise WesError("WES output exceeds size limit")
            try:
                result = json.loads(body)
            except ValueError:
                raise WesError("WES output is not valid JSON") from None
            if not isinstance(result, dict):
                raise WesError("Expected a JSON object in WES output")
            return result
        endpoint = httpx.URL(self.settings.endpoint)
        remote = httpx.URL(location)
        if (remote.scheme, remote.host, remote.port) == (endpoint.scheme, endpoint.host, endpoint.port):
            if remote.userinfo:
                raise WesError("Credentials in output URL are unsupported")
            return self._request("GET", location)
        raise WesError("Output requires site-specific staging; allow shared storage or a URL on the WES origin")


def main():
    parser = argparse.ArgumentParser(description="Cancel a retained site WES run after client interruption")
    parser.add_argument("record", type=Path)
    parser.add_argument(
        "--run-id", help="Adopt this operator-discovered run before canceling; verifies correlation tags"
    )
    args = parser.parse_args()
    runner = WesRunner(WesSettings.from_env())
    try:
        if args.run_id:
            runner.adopt(args.record, args.run_id)
        runner.cancel_record(args.record)
    finally:
        runner.close()


if __name__ == "__main__":
    main()

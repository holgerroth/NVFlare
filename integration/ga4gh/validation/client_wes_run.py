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

"""Validate a normal FedAvgRecipe with local and Client API WES execution against Toil."""

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import httpx
import numpy as np
from ga4gh_bridge import analysis
from ga4gh_bridge.report import render
from ga4gh_bridge.wes import WesRunner, WesSettings

import nvflare
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.client.in_process.api import TOPIC_ABORT, InProcessClientAPI

ROOT = Path(__file__).resolve().parents[1]


def wait_ready(process, endpoint):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Toil WES exited; inspect toil.log")
        try:
            response = httpx.get(endpoint + "/service-info", timeout=2)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            time.sleep(0.5)
    raise TimeoutError("Toil WES did not become ready")


def abort_probe(settings, out):
    marker = out / "probe-started"
    script = out / "sleep.py"
    script.write_text(f"from pathlib import Path\nimport time\nPath({str(marker)!r}).touch()\ntime.sleep(300)\n")
    api = InProcessClientAPI(task_metadata={})
    api.init()
    api.fl_model = FLModel(params={"numpy_key": [1, 2]})
    runner = WesRunner(settings)
    observed = False

    def check():
        nonlocal observed
        if marker.exists():
            observed = True
            api.event_manager.fire_event(TOPIC_ABORT, "validation abort")
        if not api.is_running():
            raise RuntimeError("NVFLARE stopped")

    try:
        try:
            runner.run(
                {"document": "{}", "script": {"class": "File", "location": "analysis.py"}},
                {"job_id": "cancel-probe", "site": "site-1", "task": "train", "round": "0"},
                {"analysis.py": script},
                check_cancel=check,
            )
        except RuntimeError:
            if not observed:
                raise
        record = json.loads(next(settings.state_dir.glob("*/run.json")).read_text())
        if record["state"] != "CANCELED":
            raise RuntimeError("Abort did not cancel the active WES workflow")
        return record["state"]
    finally:
        runner.close()


def run(out):
    if nvflare.__version__ != "2.9.0":
        raise RuntimeError("Use the released NVFLARE 2.9.0 package")
    out.mkdir(parents=True, exist_ok=False)
    toil = Path(os.environ.get("WES_TOIL_BIN") or shutil.which("toil") or "")
    if not toil.is_file():
        raise RuntimeError("Set WES_TOIL_BIN to Toil installed with its cwl and server extras")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/ga4gh/wes/v1"
    workflows = out / "toil-workflows"
    workflows.mkdir()
    service_env = {**os.environ, "PATH": str(toil.parent) + os.pathsep + os.environ["PATH"]}
    summary = {
        "scope": "client-wes",
        "rounds": 3,
        "nvflare_version": nvflare.__version__,
        "runs": {},
        "provenance": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("nvflare", "httpx", "numpy")},
            "nvflare_module": nvflare.__file__,
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(ROOT.rglob("*"))
                if p.suffix in (".py", ".cwl") and ".venv" not in p.parts and not p.name.startswith("._")
            },
        },
    }
    settings = WesSettings(endpoint, out / "wes-state", ROOT / "examples/site-workflow.cwl", output_dir=workflows)
    with (out / "toil.log").open("w") as log:
        service = subprocess.Popen(
            [
                str(toil),
                "server",
                "--bypass_celery",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "1",
                "--work_dir",
                str(workflows),
                "--opt=--no-container",
                "--opt=--maxCores=1",
                "--opt=--defaultMemory=256Mi",
                "--opt=--defaultDisk=256Mi",
            ],
            env=service_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            summary["provenance"]["wes_service_info"] = wait_ready(service, endpoint)
            for label, mode in (("local", "local"), ("client_wes", "wes")):
                env = {
                    **os.environ,
                    "GA4GH_ANALYSIS_MODE": mode,
                    "GA4GH_REMOTE_SITE": "site-1",
                    "SITE_WES_ENDPOINT": endpoint,
                    "SITE_WES_STATE_DIR": str(settings.state_dir),
                    "SITE_WES_WORKFLOW": str(settings.workflow),
                    "SITE_WES_OUTPUT_DIR": str(workflows),
                    "GA4GH_VALIDATION_TRACE_DIR": str(out / label / "traces"),
                }
                started = time.monotonic()
                with (out / f"{label}.log").open("w") as console:
                    subprocess.run(
                        [sys.executable, str(ROOT / "examples/job.py"), "--workspace", str(out / label)],
                        env=env,
                        cwd=ROOT,
                        stdout=console,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=600,
                    )
                models = list((out / label).rglob("server.npy"))
                if len(models) != 1:
                    raise RuntimeError(f"Expected one model for {label}; inspect its simulator log")
                model = np.load(models[0], allow_pickle=False)
                traces = [json.loads(p.read_text()) for p in sorted((out / label / "traces").glob("*.json"))]
                summary["runs"][label] = {
                    "model": model.tolist(),
                    "elapsed_seconds": time.monotonic() - started,
                    "job_id": traces[0]["job_id"] if traces else None,
                }
                if mode == "wes":
                    summary["site_rounds"] = traces
                np.save(out / f"{label}.npy", model)
                print(f"Completed {label}: {model.tolist()}", flush=True)
            records = [json.loads(p.read_text()) for p in sorted(settings.state_dir.glob("*/run.json"))]
            runner = WesRunner(settings)
            try:
                record = records[0]
                trace = next(t for t in summary["site_rounds"] if all(t[k] == v for k, v in record["context"].items()))
                params = {
                    "document": json.dumps({"weights": trace["input"]}),
                    "script": {"class": "File", "location": "analysis.py"},
                }
                outputs = runner.run(params, record["context"], {"analysis.py": Path(analysis.__file__)})
                summary["run_reused"] = runner.read_json_output(outputs["result"])["weights"] == trace["output"]
                summary["wes_runs"] = [
                    {
                        **r["context"],
                        "id": r["run_id"],
                        "state": runner._request("GET", f"runs/{r['run_id']}/status")["state"],
                    }
                    for r in records
                ]
            finally:
                runner.close()
            probe_settings = WesSettings(endpoint, out / "probe-state", settings.workflow, output_dir=workflows)
            summary["wes_cancel_probe"] = abort_probe(probe_settings, out)
            print("Active WES workflow canceled through NVFLARE Client API abort detection", flush=True)
        except BaseException as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(service.pid, signal.SIGTERM)
            try:
                service.wait(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(service.pid, signal.SIGKILL)
                service.wait(timeout=10)
            summary["provenance"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
            validation = render(summary, out / "report")
    if not validation["passed"]:
        raise RuntimeError("Validation failed; inspect report/validation.json")
    print(f"PASS: {out / 'report/index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out.resolve())

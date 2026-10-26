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

"""Run a private Funnel service and the synthetic POC inside a dedicated Linux Docker container."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import yaml
from ga4gh_bridge.report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", required=True, type=Path, help="Empty directory mounted at the SAME host/container path"
    )
    parser.add_argument("--label", default="Linux Docker + Funnel")
    args = parser.parse_args()
    root = args.out.resolve()
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("--out must be an existing empty dedicated output directory")
    label = "ga4gh-validation=" + uuid.uuid4().hex
    image = os.environ.get("TES_IMAGE", "python:3.12-slim")
    subprocess.run(["docker", "pull", image], check=True, timeout=300)
    config = {
        "Database": "boltdb",
        "Compute": "local",
        "EventWriters": ["boltdb", "log"],
        "Server": {"HTTPPort": "8001", "RPCPort": "9090", "DisableHTTPCache": True, "TaskAccess": "All"},
        "RPCClient": {"ServerAddress": "localhost:9090"},
        "Worker": {
            "WorkDir": str(root / "funnel-work"),
            "PollingRate": "0.5s",
            "LogUpdateRate": "1s",
            "Container": {
                "DriverCommand": "docker",
                "RunCommand": (
                    "run -i --read-only --rm --network none --cpus 1 --memory 1g "
                    f"--label {label} "
                    '{{.GetEnvArgs}} {{if .Name}}--name "{{.Name}}"{{end}} '
                    '{{if .Workdir}}--workdir "{{.Workdir}}"{{end}} '
                    '{{range .Volumes}}--volume "{{.HostPath}}:{{.ContainerPath}}:'
                    '{{if .Readonly}}ro{{else}}rw{{end}}" {{end}} {{.Image}} {{.Command}}'
                ),
            },
        },
        "BoltDB": {"Path": str(root / "funnel.db")},
        "LocalStorage": {"AllowedDirs": [str(root)]},
        "Logger": {"Level": "info"},
    }
    config_path = root / "funnel.yaml"
    config_path.write_text(yaml.safe_dump(config))
    service = None
    code = 1
    leftovers = []

    def containers():
        return subprocess.check_output(
            ["docker", "ps", "-aq", "--filter", f"label={label}"], text=True, timeout=15
        ).split()

    try:
        with (root / "funnel.log").open("w") as log:
            service = subprocess.Popen(
                ["funnel", "server", "run", "--config", str(config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 30
            while True:
                if service.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Funnel failed to start; inspect funnel.log")
                try:
                    response = httpx.get("http://127.0.0.1:8001/v1/tasks", timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.5)
            with (root / "validation.log").open("w") as validation_log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "tests/poc_smoke.py",
                        "--workdir",
                        str(root / "run"),
                        "--tes-endpoint",
                        "http://127.0.0.1:8001/v1",
                        "--label",
                        args.label,
                    ],
                    stdout=validation_log,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                )
                code = result.returncode
            deadline = time.monotonic() + 30
            while (leftovers := containers()) and time.monotonic() < deadline:
                time.sleep(1)
            report = root / "run/report"
            summary = json.loads((report / "summary.json").read_text())
            summary["provenance"].update(
                {
                    "funnel_version": subprocess.check_output(["funnel", "version"], text=True).strip(),
                    "docker_version": subprocess.check_output(
                        ["docker", "version", "--format", "{{.Client.Version}}"], text=True
                    ).strip(),
                    "tes_image": image,
                    "tes_image_id": subprocess.check_output(
                        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
                    ).strip(),
                    "tes_container_network": "none",
                    "task_container_label": label,
                    "remaining_task_containers": len(leftovers),
                    "runner_exit_code": code,
                }
            )
            if leftovers:
                summary["error"] = "TES left task containers behind; validation wrapper performed emergency cleanup"
            code = code or int(not render(summary, report)["passed"])
            (root / "exit-code.txt").write_text(str(code) + "\n")
    finally:
        # Only containers carrying this fresh run's label are owned by this wrapper.
        for container in containers():
            subprocess.run(["docker", "rm", "-f", container], check=True, timeout=30)
        if service and service.poll() is None:
            service.terminate()
            try:
                service.wait(timeout=20)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait(timeout=10)
    print(f"Validation exit code: {code}; artifacts: {root / 'run/report'}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

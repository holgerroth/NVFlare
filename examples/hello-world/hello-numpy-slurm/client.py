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

"""NVFlare client that submits one CPU-only Slurm job per training round."""

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
SCHEDULER_COMMAND_TIMEOUT = 15


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_job_id(sbatch_output: str) -> str:
    job_id = sbatch_output.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unable to parse Slurm job ID from sbatch output: {sbatch_output!r}")
    return job_id


def normalized_state(state: str) -> str:
    fields = state.split(maxsplit=1)
    return fields[0].split("+", maxsplit=1)[0] if fields else ""


def query_job(
    job_id: str, squeue: str, sacct: str
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    queue_error = None
    try:
        queue_result = subprocess.run(
            [squeue, "-h", "-j", job_id, "-o", "%T|%N"],
            check=False,
            capture_output=True,
            text=True,
            timeout=SCHEDULER_COMMAND_TIMEOUT,
        )
        if queue_result.returncode == 0 and queue_result.stdout.strip():
            state, _, nodes = queue_result.stdout.strip().partition("|")
            state = normalized_state(state)
            if state not in TERMINAL_STATES:
                return state, None, nodes, None, "squeue"
        elif queue_result.returncode != 0:
            queue_error = queue_result.stderr.strip() or f"squeue exited {queue_result.returncode}"
    except (OSError, subprocess.TimeoutExpired) as error:
        queue_error = str(error)

    try:
        accounting_result = subprocess.run(
            [sacct, "-n", "-P", "-j", job_id, "-o", "JobIDRaw,State,ExitCode,NodeList"],
            check=False,
            capture_output=True,
            text=True,
            timeout=SCHEDULER_COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, None, None, str(error), "sacct"
    if accounting_result.returncode != 0:
        error = accounting_result.stderr.strip() or f"sacct exited {accounting_result.returncode}"
        return None, None, None, error, "sacct"

    for line in accounting_result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0] == job_id:
            return normalized_state(fields[1]), fields[2], fields[3], None, "sacct"
    detail = queue_error or "job is absent from squeue and sacct"
    return None, None, None, detail, "sacct"


def cancel_job(scancel: str, job_id: str) -> None:
    try:
        result = subprocess.run(
            [scancel, job_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=SCHEDULER_COMMAND_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Unable to cancel Slurm job {job_id}: {error}")
        return
    if result.returncode != 0:
        print(f"Unable to cancel Slurm job {job_id}: {result.stderr.strip()}")


def wait_for_job(
    job_id: str,
    *,
    squeue: str,
    sacct: str,
    timeout: float,
    poll_interval: float,
) -> tuple[str, Optional[str], Optional[str]]:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if not flare.is_running():
            raise RuntimeError(f"NVFlare stopped while Slurm job {job_id} was active")

        state, exit_code, nodes, error, source = query_job(job_id, squeue, sacct)
        if error:
            last_error = error
        if state in TERMINAL_STATES and exit_code:
            print(f"Slurm job {job_id} reached {state} via {source}")
            return state, exit_code, nodes
        if state in TERMINAL_STATES:
            last_error = f"Slurm reported {state}, but accounting has not published an exit code"
        time.sleep(poll_interval)

    detail = f"; last scheduler response: {last_error}" if last_error else ""
    raise TimeoutError(f"Slurm job {job_id} did not finish within {timeout} seconds{detail}")


def _safe_directory_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)


def submit_round(input_array: np.ndarray, current_round: int, client_name: str, args: argparse.Namespace) -> np.ndarray:
    round_dir = args.slurm_work_dir / _safe_directory_name(client_name) / f"round_{current_round:03d}"
    round_dir.mkdir(parents=True, exist_ok=False)
    input_path = round_dir / "input.npy"
    output_path = round_dir / "output.npy"
    worker_metadata = round_dir / "worker_metadata.json"
    stdout_path = round_dir / "slurm.out"
    stderr_path = round_dir / "slurm.err"
    np.save(input_path, input_array)

    worker_script = Path(__file__).with_name("slurm_worker.py").resolve()
    worker_command = shlex.join(
        [
            args.worker_python or sys.executable,
            str(worker_script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--metadata",
            str(worker_metadata),
            "--learning_rate",
            str(args.learning_rate),
        ]
    )
    submit_command = [args.sbatch, "--parsable"]
    if args.slurm_account:
        submit_command.append(f"--account={args.slurm_account}")
    submit_command.extend(
        [
            f"--partition={args.slurm_partition}",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=1G",
            f"--time={args.slurm_time}",
            f"--job-name=nvf-np-r{current_round}",
            f"--chdir={round_dir}",
            f"--output={stdout_path}",
            f"--error={stderr_path}",
            "--export=NONE",
            "--wrap",
            worker_command,
        ]
    )
    (round_dir / "submit_command.json").write_text(
        json.dumps({"argv": submit_command}, indent=2) + "\n", encoding="utf-8"
    )

    submit_result = subprocess.run(submit_command, check=True, capture_output=True, text=True, timeout=30)
    job_id = parse_job_id(submit_result.stdout)
    jobs_path = args.slurm_work_dir.parent / "slurm_jobs.jsonl"
    append_jsonl(
        jobs_path,
        {
            "client": client_name,
            "event": "submitted",
            "job_id": job_id,
            "login_host": socket.gethostname(),
            "round": current_round,
            "submitted_at": utc_now(),
        },
    )
    print(f"Submitted Slurm job {job_id} for round {current_round}")

    active = True
    try:
        state, exit_code, nodes = wait_for_job(
            job_id,
            squeue=args.squeue,
            sacct=args.sacct,
            timeout=args.slurm_timeout,
            poll_interval=args.poll_interval,
        )
        active = False
    except BaseException:
        if active:
            cancel_job(args.scancel, job_id)
        raise

    append_jsonl(
        jobs_path,
        {
            "client": client_name,
            "completed_at": utc_now(),
            "event": "terminal",
            "exit_code": exit_code,
            "job_id": job_id,
            "nodes": nodes,
            "round": current_round,
            "state": state,
        },
    )
    if state != "COMPLETED" or exit_code != "0:0":
        raise RuntimeError(
            f"Slurm job {job_id} failed with state={state}, exit_code={exit_code}; "
            f"inspect {stdout_path} and {stderr_path}"
        )
    if not output_path.is_file() or not worker_metadata.is_file():
        raise RuntimeError(f"Slurm job {job_id} completed without all expected artifacts in {round_dir}")
    return np.load(output_path, allow_pickle=False)


def define_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    parser.add_argument("--slurm_work_dir", type=Path, required=True)
    parser.add_argument("--slurm_account")
    parser.add_argument("--slurm_partition", required=True)
    parser.add_argument("--slurm_time", required=True)
    parser.add_argument("--slurm_timeout", type=float, required=True)
    parser.add_argument("--poll_interval", type=float, default=1.0)
    parser.add_argument("--worker_python")
    parser.add_argument("--sbatch", default=shutil.which("sbatch"))
    parser.add_argument("--squeue", default=shutil.which("squeue"))
    parser.add_argument("--sacct", default=shutil.which("sacct"))
    parser.add_argument("--scancel", default=shutil.which("scancel"))
    args = parser.parse_args()
    for command_name in ("sbatch", "squeue", "sacct", "scancel"):
        if not getattr(args, command_name):
            raise RuntimeError(f"Required Slurm command is unavailable: {command_name}")
    return args


def main():
    args = define_parser()
    flare.init()
    client_name = flare.system_info()["site_name"]
    print(f"Client {client_name} initialized on the login node")

    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        input_array = input_model.params[NPConstants.NUMPY_KEY]
        print(f"Client {client_name}, current_round={current_round}, input_mean={float(np.mean(input_array))}")

        new_params = submit_round(input_array, current_round, client_name, args)
        metrics = {"weight_mean": float(np.mean(new_params))}
        print(f"Round {current_round} evaluation metrics: {metrics}")

        if args.update_type == "diff":
            params_to_send = new_params - input_array
            params_type = flare.ParamsType.DIFF
        else:
            params_to_send = new_params
            params_type = flare.ParamsType.FULL
        output_model = flare.FLModel(
            params={NPConstants.NUMPY_KEY: params_to_send},
            params_type=params_type,
            metrics=metrics,
            current_round=current_round,
        )
        flare.send(output_model)


if __name__ == "__main__":
    main()

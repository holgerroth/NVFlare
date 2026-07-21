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

"""Run one NVFlare client whose training rounds are delegated to Slurm."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv


def define_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=Path, required=True, help="Unique directory on storage shared with Slurm nodes"
    )
    parser.add_argument("--slurm_partition", required=True, help="CPU-only Slurm partition")
    parser.add_argument("--slurm_account", help="Slurm account; omit to use the cluster default")
    parser.add_argument(
        "--worker_python", help="Python executable visible on compute nodes; defaults to sys.executable"
    )
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--update_type", choices=["full", "diff"], default="full")
    parser.add_argument("--slurm_time", default="00:02:00")
    parser.add_argument("--slurm_timeout", type=float, default=300.0)
    parser.add_argument("--poll_interval", type=float, default=1.0)
    parser.add_argument(
        "--log_config",
        default=None,
        help="Log config mode, log config JSON file, or logging level",
    )
    return parser.parse_args()


def _require_single_cli_token(name: str, value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValueError(f"{name} cannot contain whitespace: {value!r}")
    return value


def _build_train_args(args: argparse.Namespace, run_dir: Path) -> str:
    values = [
        "--learning_rate",
        str(args.learning_rate),
        "--update_type",
        args.update_type,
        "--slurm_work_dir",
        str(run_dir / "rounds"),
        "--slurm_partition",
        args.slurm_partition,
        "--slurm_time",
        args.slurm_time,
        "--slurm_timeout",
        str(args.slurm_timeout),
        "--poll_interval",
        str(args.poll_interval),
    ]
    if args.slurm_account:
        values.extend(["--slurm_account", args.slurm_account])
    if args.worker_python:
        values.extend(["--worker_python", args.worker_python])
    return " ".join(_require_single_cli_token("training argument", value) for value in values)


def main():
    args = define_parser()
    if args.num_rounds < 1:
        raise ValueError("num_rounds must be positive")
    if args.slurm_timeout <= 0:
        raise ValueError("slurm_timeout must be positive")
    if args.poll_interval <= 0:
        raise ValueError("poll_interval must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    worker_script = Path(__file__).with_name("slurm_worker.py").resolve()
    if not worker_script.is_file():
        raise FileNotFoundError(f"Slurm worker does not exist: {worker_script}")
    train_args = _build_train_args(args, run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "learning_rate": args.learning_rate,
        "num_clients": 1,
        "num_rounds": args.num_rounds,
        "poll_interval": args.poll_interval,
        "run_dir": str(run_dir),
        "slurm_account": args.slurm_account,
        "slurm_partition": args.slurm_partition,
        "slurm_time": args.slurm_time,
        "slurm_timeout": args.slurm_timeout,
        "update_type": args.update_type,
        "worker_python": args.worker_python,
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    recipe = NumpyFedAvgRecipe(
        name="hello-numpy-slurm",
        min_clients=1,
        num_rounds=args.num_rounds,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script="client.py",
        train_args=train_args,
        params_transfer_type=TransferType.FULL if args.update_type == "full" else TransferType.DIFF,
        key_metric="weight_mean",
    )
    recipe.add_client_file(str(worker_script))

    env = SimEnv(num_clients=1, log_config=args.log_config, workspace_root=str(run_dir / "nvflare_workspace"))
    run = recipe.execute(env)
    result = run.get_result()
    status = run.get_status()
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "status": status,
    }
    (run_dir / "orchestrator_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("Result can be found in:", result)
    print("Job Status is:", status)
    print()


if __name__ == "__main__":
    main()

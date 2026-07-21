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

"""CPU-only Slurm worker for one hello-numpy training round."""

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_save_array(path: Path, value: np.ndarray) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as stream:
        np.save(stream, value)
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(path)


def atomic_save_json(path: Path, value: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def train(input_array: np.ndarray, learning_rate: float) -> np.ndarray:
    return input_array + learning_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    args = parser.parse_args()

    started_at = utc_now()
    input_array = np.load(args.input, allow_pickle=False)
    output_array = train(input_array, args.learning_rate)
    atomic_save_array(args.output, output_array)

    metadata = {
        "completed_at": utc_now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "input_mean": float(np.mean(input_array)),
        "learning_rate": args.learning_rate,
        "output_mean": float(np.mean(output_array)),
        "pid": os.getpid(),
        "python": sys.executable,
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": started_at,
    }
    atomic_save_json(args.metadata, metadata)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()

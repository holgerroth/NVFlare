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

import importlib.util
import os
import subprocess

import pytest


def _load_client_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "examples", "hello-world", "hello-numpy-slurm", "client.py")
    spec = importlib.util.spec_from_file_location("hello_numpy_slurm_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_job_id_accepts_cluster_suffix():
    client_module = _load_client_module()

    assert client_module.parse_job_id("12345;cluster\n") == "12345"


def test_parse_job_id_rejects_unexpected_output():
    client_module = _load_client_module()

    with pytest.raises(RuntimeError, match="Unable to parse Slurm job ID"):
        client_module.parse_job_id("Submitted batch job")


def test_query_job_returns_active_squeue_state(monkeypatch):
    client_module = _load_client_module()
    result = subprocess.CompletedProcess(args=[], returncode=0, stdout="RUNNING|cpu-00001\n", stderr="")
    monkeypatch.setattr(client_module.subprocess, "run", lambda *args, **kwargs: result)

    assert client_module.query_job("12345", "squeue", "sacct") == (
        "RUNNING",
        None,
        "cpu-00001",
        None,
        "squeue",
    )


def test_query_job_uses_top_level_sacct_record(monkeypatch):
    client_module = _load_client_module()
    results = iter(
        [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="12345.batch|COMPLETED|0:0|cpu-00001\n12345|COMPLETED+|0:0|cpu-00001\n",
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(client_module.subprocess, "run", lambda *args, **kwargs: next(results))

    assert client_module.query_job("12345", "squeue", "sacct") == (
        "COMPLETED",
        "0:0",
        "cpu-00001",
        None,
        "sacct",
    )

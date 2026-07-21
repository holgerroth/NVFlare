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
from types import SimpleNamespace

import pytest


def _load_job_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "examples", "hello-world", "hello-numpy-slurm", "job.py")
    spec = importlib.util.spec_from_file_location("hello_numpy_slurm_job", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_train_args_includes_optional_slurm_values(tmp_path):
    job_module = _load_job_module()
    args = SimpleNamespace(
        learning_rate=1.0,
        update_type="full",
        slurm_partition="cpu_short",
        slurm_time="00:02:00",
        slurm_timeout=300.0,
        poll_interval=1.0,
        slurm_account="test_account",
        worker_python="/shared/venv/bin/python",
    )

    train_args = job_module._build_train_args(args, tmp_path)

    assert "--slurm_account test_account" in train_args
    assert "--worker_python /shared/venv/bin/python" in train_args
    assert f"--slurm_work_dir {tmp_path / 'rounds'}" in train_args


def test_build_train_args_rejects_whitespace_in_shared_path(tmp_path):
    job_module = _load_job_module()
    args = SimpleNamespace(
        learning_rate=1.0,
        update_type="full",
        slurm_partition="cpu_short",
        slurm_time="00:02:00",
        slurm_timeout=300.0,
        poll_interval=1.0,
        slurm_account=None,
        worker_python=None,
    )

    with pytest.raises(ValueError, match="cannot contain whitespace"):
        job_module._build_train_args(args, tmp_path / "path with spaces")


def test_recipe_bundles_worker_with_public_api(tmp_path, monkeypatch):
    job_module = _load_job_module()
    run_dir = tmp_path / "run"
    args = SimpleNamespace(
        run_dir=run_dir,
        slurm_partition="cpu_short",
        slurm_account=None,
        worker_python=None,
        num_rounds=1,
        learning_rate=1.0,
        update_type="full",
        slurm_time="00:02:00",
        slurm_timeout=300.0,
        poll_interval=1.0,
        log_config=None,
    )
    added_files = []
    recipe_kwargs = {}

    class FakeRecipe:
        def add_client_file(self, file_path):
            added_files.append(file_path)

        def execute(self, env):
            raise RuntimeError("stop after recipe construction")

    monkeypatch.setattr(job_module, "define_parser", lambda: args)

    def make_recipe(**kwargs):
        recipe_kwargs.update(kwargs)
        return FakeRecipe()

    monkeypatch.setattr(job_module, "NumpyFedAvgRecipe", make_recipe)

    with pytest.raises(RuntimeError, match="stop after recipe construction"):
        job_module.main()

    assert len(added_files) == 1
    assert added_files[0].endswith("slurm_worker.py")
    assert recipe_kwargs["key_metric"] == "weight_mean"

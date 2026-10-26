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

import copy

import numpy as np
import pytest
from ga4gh_bridge.report import compare, render, validate


@pytest.fixture
def evidence():
    initial = np.arange(1, 10).reshape(3, 3)
    return {
        "scope": "end-to-end",
        "rounds": 3,
        "runs": {
            name: {"model": (initial + 3).tolist(), "elapsed_seconds": 1, "job_id": name}
            for name in ("direct", "wes", "mixed")
        },
        "site_rounds": [
            {
                "site": site,
                "mode": mode,
                "round": str(index),
                "input": (initial + index).tolist(),
                "output": (initial + index + 1).tolist(),
            }
            for index in range(3)
            for site, mode in (("site-1", "tes"), ("site-2", "local"))
        ],
        "tes_tasks": [
            {"id": str(index), "state": "COMPLETE", "job_id": "mixed", "site": "site-1", "round": str(index)}
            for index in range(3)
        ],
        "cancellation": {"early": "CANCELED", "running": "CANCELED"},
        "restart_recovered": True,
        "tes_protocol_fixture": False,
        "tes_cancel_probe": "CANCELED",
    }


@pytest.mark.parametrize(
    "bad",
    [None, [[1]], [1, 2, 3], [[True] * 3] * 3, [["4"] * 3] * 3, [[float("nan")] * 3] * 3, [[float("inf")] * 3] * 3],
)
def test_rejects_invalid_arrays(bad):
    assert not compare(bad, np.arange(1, 10).reshape(3, 3) + 3)["passed"]


def test_detects_shared_wrong_result(evidence):
    for run in evidence["runs"].values():
        run["model"][0][0] += 0.01
    result = validate(evidence)
    assert not result["passed"]
    assert sum(not c["passed"] for c in result["checks"]) == 3


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e["site_rounds"].pop(),
        lambda e: e["site_rounds"].append(copy.deepcopy(e["site_rounds"][0])),
        lambda e: e["site_rounds"][0].update(mode="local"),
        lambda e: e["site_rounds"][1]["output"][0].__setitem__(0, -99),
        lambda e: e["tes_tasks"][0].update(state="RUNNING"),
        lambda e: e["tes_tasks"][0].update(job_id="wrong-job"),
        lambda e: e.update(restart_recovered=False),
        lambda e: e.update(tes_cancel_probe="CANCEL_UNCONFIRMED"),
        lambda e: e.update(error="runner failed"),
        lambda e: e["runs"].pop("mixed"),
    ],
)
def test_fails_incomplete_or_inconsistent_evidence(evidence, change):
    assert validate(evidence)["passed"]
    change(evidence)
    assert not validate(evidence)["passed"]


def test_report_is_portable_and_escapes_metadata(evidence, tmp_path):
    pytest.importorskip("matplotlib")
    evidence["provenance"] = {"label": "<script>alert(1)</script>"}
    assert render(evidence, tmp_path)["passed"]
    document = (tmp_path / "index.html").read_text()
    assert "data:image/png;base64," in document
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert (tmp_path / "arrays.svg").is_file()
    assert (tmp_path / "rounds-and-timing.png").is_file()


def test_failed_report_retains_failure(evidence, tmp_path):
    pytest.importorskip("matplotlib")
    evidence["runs"].pop("wes")
    assert not render(evidence, tmp_path)["passed"]
    assert "FAIL" in (tmp_path / "index.html").read_text()


@pytest.mark.parametrize("damage", ["wrong_mode", "extra_run", "wrong_job", "wrong_output", "reuse", "cancel"])
def test_client_wes_report_rejects_incomplete_evidence(evidence, damage):
    evidence["scope"] = "client-wes"
    evidence["runs"] = {"local": evidence["runs"]["direct"], "client_wes": evidence["runs"]["mixed"]}
    for trace in evidence["site_rounds"]:
        if trace["site"] == "site-1":
            trace["mode"] = "wes"
    evidence["wes_runs"] = evidence.pop("tes_tasks")
    evidence.update(run_reused=True, wes_cancel_probe="CANCELED")
    assert validate(evidence)["passed"]
    if damage == "wrong_mode":
        evidence["site_rounds"][0]["mode"] = "tes"
    elif damage == "extra_run":
        evidence["wes_runs"].append(copy.deepcopy(evidence["wes_runs"][0]))
    elif damage == "wrong_job":
        evidence["wes_runs"][0]["job_id"] = "another-job"
    elif damage == "wrong_output":
        evidence["runs"]["client_wes"]["model"][0][0] += 1
    elif damage == "reuse":
        evidence["run_reused"] = False
    else:
        evidence["wes_cancel_probe"] = "COMPLETE"
    assert not validate(evidence)["passed"]

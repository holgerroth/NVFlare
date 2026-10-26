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

import io
import json
import stat
import zipfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from ga4gh_bridge.app import create_app
from ga4gh_bridge.backend import PermanentBackendError
from ga4gh_bridge.config import Settings
from ga4gh_bridge.service import STATES
from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

PREFIX = "/ga4gh/wes/v1"
TOKEN = "test-pilot-token-with-32-characters"


class FakeBackend:
    def __init__(self):
        self.jobs = {}
        self.calls = []
        self.status = "RUNNING"
        self.unavailable = False
        self.lose_submit_reply = False
        self.missing_output = False
        self.cancel_calls = []
        self.on_submit = None

    def submit(self, job_dir, token):
        self.calls.append(token)
        job_id = self.jobs.setdefault(token, f"job-{len(self.jobs)}")
        if self.on_submit:
            self.on_submit()
        if self.lose_submit_reply:
            self.lose_submit_reply = False
            raise ConnectionError("Authorization: Bearer secret-must-not-appear")
        return job_id

    def meta(self, job_id):
        if self.unavailable:
            raise ConnectionError("secret-must-not-appear")
        return {"status": self.status}

    def cancel(self, job_id):
        self.cancel_calls.append(job_id)
        self.status = "FINISHED:ABORTED"

    def download(self, job_id, destination):
        root = destination / job_id
        root.mkdir()
        if not self.missing_output:
            (root / "model.npy").write_bytes(b"model")
        (root / "private-key.pem").write_text("secret-must-not-appear")
        return root

    def close(self):
        pass


def bundle(extra=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("job/meta.json", json.dumps({"deploy_map": {"app": ["server", "site-1"]}}))
        archive.writestr("job/app/config/config_fed_server.json", "{}")
        for name, value in extra or []:
            archive.writestr(name, value)
    return stream.getvalue()


def submit(client, content=None, **fields):
    data = {"workflow_type": "NVFLARE", "workflow_type_version": "1.0", "workflow_url": "job.zip"}
    data.update(fields)
    return client.post(PREFIX + "/runs", data=data, files={"workflow_attachment": ("job.zip", content or bundle())})


@pytest.fixture
def settings(tmp_path):
    return Settings(tmp_path, TOKEN, output_paths=("model.npy",), result_attempts=2)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def client(settings, backend):
    with TestClient(create_app(settings, backend, background=False)) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def contract(name, instance):
    root = Path(__file__).parent / "specs"
    wes = yaml.safe_load((root / "wes-1.1.0.yaml").read_text())
    info = yaml.safe_load((root / "service-info-1.0.0.yaml").read_text())
    uri = "https://raw.githubusercontent.com/ga4gh-discovery/ga4gh-service-info/v1.0.0/service-info.yaml"
    registry = Registry().with_resources([("urn:wes", Resource(wes, DRAFT7)), (uri, Resource(info, DRAFT7))])
    Draft7Validator({"$ref": f"urn:wes#/components/schemas/{name}"}, registry=registry).validate(instance)


def test_all_wes_response_contracts_and_output_allowlist(client, backend):
    info = client.get(PREFIX + "/service-info").json()
    contract("ServiceInfo", info)
    assert info["workflow_engine_versions"]["NVFLARE"]["workflow_engine_version"] == ["2.9.0"]
    response = submit(client, workflow_engine="NVFLARE", workflow_engine_version="2.9.0")
    assert response.status_code == 200
    contract("RunId", response.json())
    rid = response.json()["run_id"]
    base = f"{PREFIX}/runs/{rid}"
    svc = client.app.state.service
    svc.tick()
    contract("RunStatus", client.get(base + "/status").json())
    contract("RunListResponse", client.get(PREFIX + "/runs").json())
    contract("RunLog", client.get(base).json())
    contract("TaskListResponse", client.get(base + "/tasks").json())
    contract("TaskLog", client.get(base + "/tasks/federation").json())
    assert client.get(base + "/outputs/0").status_code == 404
    backend.status = "FINISHED:COMPLETED"
    svc.tick()
    log = client.get(base).json()
    contract("RunLog", log)
    assert log["state"] == "COMPLETE"
    assert set(log["outputs"]) == {"model.npy"}
    assert client.get(base + "/outputs/0").content == b"model"
    assert client.get(base + "/outputs/1").status_code == 404
    contract("RunId", client.post(base + "/cancel").json())
    assert client.get(base + "/status").json()["state"] == "COMPLETE"


@pytest.mark.parametrize(
    "status,expected", list(STATES.items()) + [("FINISHED:ABORTED", "EXECUTOR_ERROR"), ("NEW_UNKNOWN_STATE", "UNKNOWN")]
)
def test_status_mapping(client, backend, status, expected):
    rid = submit(client).json()["run_id"]
    backend.status = status
    client.app.state.service.tick()
    assert client.get(f"{PREFIX}/runs/{rid}/status").json()["state"] == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"workflow_type": "CWL"},
        {"workflow_type_version": "2"},
        {"workflow_params": '{"a":1}'},
        {"workflow_params": "[]"},
        {"tags": "[]"},
        {"tags": '{"a":1}'},
        {"tags": "bad-json"},
        {"workflow_url": "https://example.org/job.zip"},
        {"workflow_url": "../job.zip"},
        {"workflow_engine": "other"},
        {"workflow_engine_version": "bad"},
        {"unexpected": "x"},
    ],
)
def test_reject_unsupported_requests(client, fields):
    response = submit(client, **fields)
    assert response.status_code == 400
    contract("ErrorResponse", response.json())
    assert not client.app.state.service.store.all()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "job/../../escape", "job\\escape", "job/C:escape"])
def test_reject_archive_traversal(client, path):
    assert submit(client, bundle([(path, "bad")])).status_code == 400


def test_reject_symlink_duplicate_corrupt_and_oversized_archives(client, settings):
    link = zipfile.ZipInfo("job/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert submit(client, bundle([(link, "/etc/passwd")])).status_code == 400
    assert submit(client, bundle([("JOB/META.JSON", "{}")])).status_code == 400
    assert submit(client, b"broken").status_code == 400
    object.__setattr__(settings, "max_unpacked_bytes", 10)
    assert submit(client).status_code == 400


def test_authentication_includes_output_and_diagnostics(client):
    rid = submit(client).json()["run_id"]
    del client.headers["Authorization"]
    for path in ("/service-info", "/runs", f"/runs/{rid}/diagnostics", f"/runs/{rid}/outputs/0"):
        response = client.get(PREFIX + path)
        assert response.status_code == 401
        contract("ErrorResponse", response.json())
    assert submit(client).status_code == 401
    response = client.get(PREFIX + "/service-info", headers={b"Authorization": b"Bearer \xff"})
    assert response.status_code == 401


def test_streaming_upload_limit(tmp_path, backend):
    settings = Settings(tmp_path, TOKEN, max_upload_bytes=100)
    with TestClient(create_app(settings, backend, background=False)) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        response = client.post(PREFIX + "/runs", content=iter([b"x" * 60, b"y" * 60]))
        assert response.status_code == 413


@pytest.mark.parametrize("stage", ["before_submit", "running", "during_submit", "completed"])
def test_cancellation_races(client, backend, stage):
    rid = submit(client).json()["run_id"]
    svc = client.app.state.service
    if stage == "during_submit":
        backend.on_submit = lambda: svc.cancel(rid)
        svc.tick()
    else:
        if stage != "before_submit":
            svc.tick()
        if stage == "completed":
            backend.status = "FINISHED:COMPLETED"
        client.post(f"{PREFIX}/runs/{rid}/cancel")
    svc.tick()
    svc.tick()
    state = svc.store.get(rid)["state"]
    assert state == ("COMPLETE" if stage == "completed" else "CANCELED")
    if stage == "before_submit":
        assert not backend.calls


def test_restart_after_ambiguous_submission_and_cancel(settings, backend):
    backend.lose_submit_reply = True
    with TestClient(create_app(settings, backend, background=False)) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        rid = submit(client).json()["run_id"]
        client.app.state.service.tick()
        client.post(f"{PREFIX}/runs/{rid}/cancel")
    with TestClient(create_app(settings, backend, background=False)) as client:
        svc = client.app.state.service
        svc.tick()
        svc.tick()
        assert svc.store.get(rid)["state"] == "CANCELED"
    assert backend.calls == [rid, rid]
    assert len(backend.jobs) == 1
    assert backend.cancel_calls == ["job-0"]


def test_disconnect_is_not_terminal_or_secret_leak(client, backend):
    rid = submit(client).json()["run_id"]
    backend.unavailable = True
    client.app.state.service.tick()
    response = client.get(f"{PREFIX}/runs/{rid}/diagnostics")
    assert "secret-must-not-appear" not in response.text
    assert response.json()["state"] == "QUEUED"
    backend.unavailable = False
    client.app.state.service.tick()
    assert client.app.state.service.store.get(rid)["state"] == "RUNNING"


def test_result_failure_is_retried_and_never_complete(client, backend):
    rid = submit(client).json()["run_id"]
    backend.status = "FINISHED:COMPLETED"
    backend.missing_output = True
    svc = client.app.state.service
    svc.tick()
    assert svc.store.get(rid)["state"] not in ("COMPLETE", "SYSTEM_ERROR")
    svc.tick()
    assert svc.store.get(rid)["state"] == "SYSTEM_ERROR"
    assert not svc.run_log(rid)["outputs"]


def test_permanent_rejection(client, backend):
    def reject(*args):
        raise PermanentBackendError("private backend detail")

    backend.submit = reject
    rid = submit(client).json()["run_id"]
    client.app.state.service.tick()
    assert client.app.state.service.store.get(rid)["state"] == "SYSTEM_ERROR"


def test_pagination_is_a_snapshot_and_tokens_are_validated(client):
    ids = [submit(client).json()["run_id"] for _ in range(3)]
    page = client.get(PREFIX + "/runs?page_size=1").json()
    token = page["next_page_token"]
    submit(client)
    client.app.state.service.tick()
    page2 = client.get(PREFIX + "/runs", params={"page_size": 10, "page_token": token}).json()
    assert [r["run_id"] for r in page2["runs"]] == ids[1:]
    assert all(r["state"] == "QUEUED" for r in page2["runs"])
    assert not page2["next_page_token"]
    assert client.get(PREFIX + "/runs", params={"page_token": token + "x"}).status_code == 400
    assert client.get(PREFIX + "/runs?page_size=0").status_code == 400
    assert client.get(PREFIX + "/runs/missing").status_code == 404


def test_only_one_process_owns_state(client, settings, backend):
    with pytest.raises(RuntimeError, match="already has an owner"):
        with TestClient(create_app(settings, backend, background=False)):
            pass


def test_background_monitor_and_completed_output_survive_restart(settings, backend):
    import time

    object.__setattr__(settings, "poll_interval", 0.01)
    backend.status = "FINISHED:COMPLETED"
    with TestClient(create_app(settings, backend)) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        rid = submit(client).json()["run_id"]
        deadline = time.monotonic() + 3
        while client.get(f"{PREFIX}/runs/{rid}/status").json()["state"] != "COMPLETE":
            assert time.monotonic() < deadline
            time.sleep(0.01)
    changed = Settings(settings.state_dir, TOKEN, output_paths=("different.npy",))
    with TestClient(create_app(changed, backend, background=False)) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        assert set(client.get(f"{PREFIX}/runs/{rid}").json()["outputs"]) == {"model.npy"}
        assert client.get(f"{PREFIX}/runs/{rid}/outputs/0").content == b"model"
    assert len(backend.calls) == 1


def test_output_size_limit_blocks_publication(client, backend, settings):
    object.__setattr__(settings, "max_output_bytes", 1)
    backend.status = "FINISHED:COMPLETED"
    rid = submit(client).json()["run_id"]
    client.app.state.service.tick()
    client.app.state.service.tick()
    assert client.get(f"{PREFIX}/runs/{rid}/status").json()["state"] == "SYSTEM_ERROR"
    assert client.get(f"{PREFIX}/runs/{rid}/outputs/0").status_code == 404


def test_output_symlink_is_not_published(client, backend, tmp_path):
    def download(job_id, destination):
        secret = tmp_path / "outside.txt"
        secret.write_text("private")
        (destination / "model.npy").symlink_to(secret)
        return destination

    backend.download = download
    backend.status = "FINISHED:COMPLETED"
    rid = submit(client).json()["run_id"]
    client.app.state.service.tick()
    client.app.state.service.tick()
    assert client.app.state.service.store.get(rid)["state"] == "SYSTEM_ERROR"


def test_unknown_backend_status_is_not_reflected_in_diagnostics(client, backend):
    backend.status = "FINISHED:SECRET"
    rid = submit(client).json()["run_id"]
    client.app.state.service.tick()
    assert "SECRET" not in client.get(f"{PREFIX}/runs/{rid}/diagnostics").text

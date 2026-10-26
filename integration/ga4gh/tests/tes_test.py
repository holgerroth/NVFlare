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

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import pytest
import yaml
from ga4gh_bridge.analysis import calculate
from ga4gh_bridge.tes import AmbiguousSubmission, Cancelled, TesError, TesRunner, TesSettings
from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

CONTEXT = {"job_id": "job-1", "site": "site-1", "task": "train", "round": "0"}
INPUT = {"weights": [[1, 2], [3, 4]]}


class TesTransport:
    def __init__(self):
        self.task = None
        self.posts = 0
        self.cancelled = False
        self.status = "COMPLETE"
        self.ambiguous = False
        self.missing = False
        self.fail_get = False

    def __call__(self, request):
        if request.method == "POST" and request.url.path.endswith("/tasks"):
            self.posts += 1
            self.task = json.loads(request.content)
            output = Path(unquote(urlparse(self.task["outputs"][0]["url"]).path))
            if not self.missing:
                output.write_text(json.dumps(calculate(INPUT)))
            if self.ambiguous:
                raise httpx.ReadTimeout("private credentials", request=request)
            return httpx.Response(200, json={"id": "task-1"})
        if request.method == "POST":
            assert request.url.path.endswith("/tasks/task-1:cancel")
            self.cancelled = True
            return httpx.Response(200, json={})
        if request.url.path.endswith("/tasks"):
            assert request.url.params["tag_key"] == "nvflare_operation"
            return httpx.Response(200, json={"tasks": [{"id": "task-1"}] if self.task else []})
        if self.fail_get:
            self.fail_get = False
            raise httpx.ReadTimeout("private credentials", request=request)
        return httpx.Response(200, json={"id": "task-1", "state": "CANCELED" if self.cancelled else self.status})


@pytest.fixture
def transport():
    return TesTransport()


@pytest.fixture
def runner(tmp_path, transport):
    settings = TesSettings("http://test/ga4gh/tes/v1", tmp_path, poll_interval=0.001, cancel_timeout=0.05)
    client = httpx.Client(base_url=settings.endpoint + "/", transport=httpx.MockTransport(transport))
    runner = TesRunner(settings, client)
    yield runner
    runner.close()


def test_tes_matches_local_and_request_matches_official_schema(runner, transport):
    assert runner.run(INPUT, CONTEXT) == calculate(INPUT)
    schema = yaml.safe_load((Path(__file__).parent / "specs/tes-1.1.0.yaml").read_text())
    registry = Registry().with_resource("urn:tes", Resource(schema, DRAFT7))
    Draft7Validator({"$ref": "urn:tes#/components/schemas/tesTask"}, registry=registry).validate(transport.task)
    assert transport.task["tags"]["site"] == "site-1"
    assert transport.posts == 1
    assert not transport.cancelled
    # Repeating the same operation reuses its task; it never submits a duplicate.
    assert runner.run(INPUT, CONTEXT) == calculate(INPUT)
    assert transport.posts == 1


def test_ambiguous_submission_recovers_by_tag_without_reposting(runner, transport):
    transport.ambiguous = True
    with pytest.raises(AmbiguousSubmission, match="outcome unknown"):
        runner.run(INPUT, CONTEXT)
    assert json.loads(next(runner.root.glob("*/task.json")).read_text())["state"] == "SUBMISSION_UNKNOWN"
    assert runner.run(INPUT, CONTEXT) == calculate(INPUT)
    assert transport.posts == 1


def test_unresolved_submission_fails_loud_and_can_be_cancelled_later(runner, transport):
    transport.ambiguous = True
    with pytest.raises(AmbiguousSubmission):
        runner.run(INPUT, CONTEXT)
    task = transport.task
    transport.task = None
    with pytest.raises(AmbiguousSubmission, match="unresolved"):
        runner.run(INPUT, CONTEXT)
    assert transport.posts == 1
    transport.task = task
    transport.status = "RUNNING"
    runner.cancel_record(next(runner.root.glob("*/task.json")))
    assert transport.cancelled


@pytest.mark.parametrize("state", ["EXECUTOR_ERROR", "SYSTEM_ERROR", "CANCELED", "PREEMPTED"])
def test_remote_failures_do_not_return_results(runner, transport, state):
    transport.status = state
    with pytest.raises(TesError, match=state):
        runner.run(INPUT, CONTEXT)


def test_missing_output_fails(runner, transport):
    transport.missing = True
    with pytest.raises(TesError, match="output is missing"):
        runner.run(INPUT, CONTEXT)


@pytest.mark.parametrize("state,deadline", [("QUEUED", "queue_timeout"), ("RUNNING", "execution_timeout")])
def test_deadline_cancels_and_confirms_remote_task(runner, transport, state, deadline):
    transport.status = state
    object.__setattr__(runner.settings, deadline, 0.005)
    with pytest.raises(TesError, match="deadline exceeded"):
        runner.run(INPUT, CONTEXT)
    assert transport.cancelled
    assert json.loads(next(runner.root.glob("*/task.json")).read_text())["state"] == "CANCELED"


def test_polling_failure_cancels_remote_work_and_hides_response(runner, transport):
    transport.status = "RUNNING"
    transport.fail_get = True
    with pytest.raises(TesError) as error:
        runner.run(INPUT, CONTEXT)
    assert "private credentials" not in str(error.value)
    assert transport.cancelled


def test_abort_cancels_remote_work(runner, transport):
    transport.status = "RUNNING"

    def check():
        if transport.task:
            raise Cancelled("NVFLARE abort")

    with pytest.raises(Cancelled):
        runner.run(INPUT, CONTEXT, check_cancel=check)
    assert transport.cancelled


def test_actual_in_process_client_api_observes_abort_during_current_task(runner, transport):
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.client.in_process.api import TOPIC_ABORT, InProcessClientAPI

    api = InProcessClientAPI(task_metadata={})
    api.init()
    api.fl_model = FLModel(params={"numpy_key": [1, 2]})
    assert api.is_running()
    transport.status = "RUNNING"

    def check():
        if transport.task:
            api.event_manager.fire_event(TOPIC_ABORT, "test abort")
        try:
            if not api.is_running():
                raise Cancelled("stopped")
        except RuntimeError:
            raise Cancelled("aborted") from None

    with pytest.raises(Cancelled):
        runner.run(INPUT, CONTEXT, check_cancel=check)
    assert transport.cancelled


def test_cancellation_failure_keeps_actionable_record(runner, transport):
    transport.status = "RUNNING"
    object.__setattr__(runner.settings, "queue_timeout", 0.001)
    original = runner._request

    def fail_cancel(method, path, **kwargs):
        if path.endswith(":cancel"):
            raise TesError("unreachable")
        return original(method, path, **kwargs)

    runner._request = fail_cancel
    with pytest.raises(TesError, match="cancellation unconfirmed"):
        runner.run(INPUT, CONTEXT)
    assert json.loads(next(runner.root.glob("*/task.json")).read_text())["state"] == "CANCEL_UNCONFIRMED"


def test_same_operation_rejects_changed_input(runner):
    runner.run(INPUT, CONTEXT)
    with pytest.raises(TesError, match="different inputs"):
        runner.run({"weights": [10]}, CONTEXT)


def test_abort_on_resume_cancels_previously_submitted_task(runner, transport):
    transport.ambiguous = True
    with pytest.raises(AmbiguousSubmission):
        runner.run(INPUT, CONTEXT)
    transport.status = "RUNNING"

    def abort():
        raise Cancelled("aborted before resume")

    with pytest.raises(Cancelled):
        runner.run(INPUT, CONTEXT, check_cancel=abort)
    assert transport.cancelled

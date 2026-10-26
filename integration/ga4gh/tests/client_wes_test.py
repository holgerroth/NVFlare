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
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

import httpx
import pytest
import yaml
from ga4gh_bridge.wes import AmbiguousSubmission, WesError, WesRunner, WesSettings
from jsonschema import Draft7Validator

CONTEXT = {"job_id": "job-1", "site": "site-1", "task": "train", "round": "0"}
INPUT = {"document": '{"weights": [[1, 2]]}', "script": {"class": "File", "location": "analysis.py"}}


class Transport:
    def __init__(self, root):
        self.posts = 0
        self.form = {}
        self.state = "COMPLETE"
        self.ambiguous = False
        self.cancelled = False
        self.fail_cancel = False
        self.root = root
        self.output = root / "output.json"
        self.output.write_text('{"weights": [[2, 3]]}')

    def __call__(self, request):
        if request.method == "POST" and request.url.path.endswith("/runs"):
            self.posts += 1
            content_type = request.headers["content-type"]
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {content_type}\r\n\r\n".encode() + request.read()
            )
            self.files = {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    self.files[part.get_filename()] = part.get_payload(decode=True)
                else:
                    self.form[name] = part.get_payload(decode=True).decode()
            if self.ambiguous:
                raise httpx.ReadTimeout("private token", request=request)
            return httpx.Response(200, json={"run_id": "run-1"})
        if request.method == "POST":
            assert request.url.path.endswith("/runs/run-1/cancel")
            if self.fail_cancel:
                return httpx.Response(503, text="private diagnostics")
            self.cancelled = True
            return httpx.Response(200, json={"run_id": "run-1"})
        result = {"run_id": "run-1", "state": "CANCELED" if self.cancelled else self.state}
        if not request.url.path.endswith("/status"):
            result.update(
                request={"tags": json.loads(self.form["tags"])},
                outputs={"result": {"class": "File", "location": self.output.as_uri()}},
            )
        return httpx.Response(200, json=result)


@pytest.fixture
def configured(tmp_path):
    workflow = Path(__file__).parents[1] / "examples/site-workflow.cwl"
    settings = WesSettings(
        "http://test/ga4gh/wes/v1",
        tmp_path / "state",
        workflow,
        output_dir=tmp_path,
        poll_interval=0.001,
        cancel_timeout=0.02,
    )
    transport = Transport(tmp_path)
    client = httpx.Client(base_url=settings.endpoint + "/", transport=httpx.MockTransport(transport))
    runner = WesRunner(settings, client)
    yield runner, transport
    runner.close()


def test_submit_workflow_parameters_attachments_and_reuse(configured):
    runner, transport = configured
    script = Path(__file__).parents[1] / "ga4gh_bridge/analysis.py"
    outputs = runner.run(INPUT, CONTEXT, {"analysis.py": script})
    assert runner.read_json_output(outputs["result"]) == {"weights": [[2, 3]]}
    assert transport.form["workflow_type"] == "CWL"
    assert transport.form["workflow_type_version"] == "v1.2"
    assert transport.form["workflow_url"] == "workflow.cwl"
    assert json.loads(transport.form["workflow_params"]) == INPUT
    request = {**transport.form, "workflow_params": INPUT, "tags": json.loads(transport.form["tags"])}
    spec = yaml.safe_load((Path(__file__).parent / "specs/wes-1.1.0.yaml").read_text())
    Draft7Validator(spec["components"]["schemas"]["RunRequest"]).validate(request)
    assert set(transport.files) == {"workflow.cwl", "analysis.py"}
    assert json.loads(transport.form["tags"])["job_id"] == "job-1"
    runner.run(INPUT, CONTEXT, {"analysis.py": script})
    assert transport.posts == 1


def test_lost_response_never_reposts_and_operator_can_recover(configured):
    runner, transport = configured
    transport.ambiguous = True
    with pytest.raises(AmbiguousSubmission):
        runner.run(INPUT, CONTEXT)
    with pytest.raises(AmbiguousSubmission):
        runner.run(INPUT, CONTEXT)
    assert transport.posts == 1
    record = next(runner.root.glob("*/run.json"))
    runner.adopt(record, "run-1")
    assert runner.run(INPUT, CONTEXT)["result"]
    assert transport.posts == 1


def test_adoption_checks_tags(configured):
    runner, transport = configured
    transport.ambiguous = True
    with pytest.raises(AmbiguousSubmission):
        runner.run(INPUT, CONTEXT)
    transport.form["tags"] = "{}"
    with pytest.raises(WesError, match="correlation"):
        runner.adopt(next(runner.root.glob("*/run.json")), "unrelated-run")


@pytest.mark.parametrize("state", ["EXECUTOR_ERROR", "SYSTEM_ERROR", "CANCELED"])
def test_remote_failure(configured, state):
    runner, transport = configured
    transport.state = state
    with pytest.raises(WesError, match=state):
        runner.run(INPUT, CONTEXT)


@pytest.mark.parametrize("state,field", [("QUEUED", "queue_timeout"), ("RUNNING", "execution_timeout")])
def test_deadline_cancels_remote_work(configured, state, field):
    runner, transport = configured
    transport.state = state
    object.__setattr__(runner.settings, field, 0.005)
    with pytest.raises(WesError, match="deadline"):
        runner.run(INPUT, CONTEXT)
    assert transport.cancelled


def test_client_abort_during_polling(configured):
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.client.in_process.api import TOPIC_ABORT, InProcessClientAPI

    runner, transport = configured
    transport.state = "RUNNING"
    api = InProcessClientAPI(task_metadata={})
    api.init()
    api.fl_model = FLModel(params={"numpy_key": [1, 2]})

    def check():
        if transport.posts:
            api.event_manager.fire_event(TOPIC_ABORT, "test abort")
        if not api.is_running():
            raise RuntimeError("stopped")

    with pytest.raises(RuntimeError):
        runner.run(INPUT, CONTEXT, check_cancel=check)
    assert transport.cancelled


def test_unconfirmed_cancel_retains_record(configured):
    runner, transport = configured
    transport.state = "RUNNING"
    transport.fail_cancel = True
    object.__setattr__(runner.settings, "queue_timeout", 0.001)
    with pytest.raises(WesError, match="cancellation unconfirmed"):
        runner.run(INPUT, CONTEXT)
    record = json.loads(next(runner.root.glob("*/run.json")).read_text())
    assert record["state"] == "CANCEL_UNCONFIRMED"


def test_changed_input_and_code_cannot_reuse_run(configured, tmp_path):
    runner, _ = configured
    script = tmp_path / "analysis.py"
    script.write_text("version 1")
    runner.run(INPUT, CONTEXT, {"analysis.py": script})
    script.write_text("version 2")
    with pytest.raises(WesError, match="different inputs or workflow code"):
        runner.run(INPUT, CONTEXT, {"analysis.py": script})


def test_record_endpoint_cannot_change(configured):
    runner, _ = configured
    runner.run(INPUT, CONTEXT)
    object.__setattr__(runner.settings, "endpoint", "https://another.example/wes")
    with pytest.raises(WesError, match="another WES"):
        runner.run(INPUT, CONTEXT)


def test_abort_on_resume_cancels_existing_workflow(configured):
    runner, transport = configured
    runner.run(INPUT, CONTEXT)
    transport.state = "RUNNING"

    def abort():
        raise RuntimeError("aborted before resume")

    with pytest.raises(RuntimeError, match="aborted before resume"):
        runner.run(INPUT, CONTEXT, check_cancel=abort)
    assert transport.cancelled
    assert transport.posts == 1


def test_output_rejects_unapproved_hosts_paths_and_symlinks(configured, tmp_path):
    runner, transport = configured
    for location in ["https://other.example/result", "file:///etc/passwd"]:
        with pytest.raises(WesError):
            runner.read_json_output(location)
    link = tmp_path / "link.json"
    link.symlink_to(transport.output)
    with pytest.raises(WesError, match="Symlink"):
        runner.read_json_output(link.as_uri())


def test_missing_malformed_and_oversized_outputs(configured):
    runner, transport = configured
    for body in ["invalid", "[]", "x" * (runner.settings.max_bytes + 1)]:
        transport.output.write_text(body)
        with pytest.raises(WesError):
            runner.read_json_output(transport.output.as_uri())
    transport.output.unlink()
    with pytest.raises(WesError):
        runner.read_json_output(transport.output.as_uri())


def test_output_download_uses_same_origin_token_and_no_redirects(configured):
    runner, _ = configured

    def redirect(request):
        assert request.headers["authorization"] == "Bearer site-secret"
        return httpx.Response(302, headers={"location": "https://other.example"})

    runner.client.close()
    runner.client = httpx.Client(
        headers={"Authorization": "Bearer site-secret"}, transport=httpx.MockTransport(redirect)
    )
    with pytest.raises(WesError) as error:
        runner.read_json_output("http://test/output.json")
    assert "site-secret" not in str(error.value)

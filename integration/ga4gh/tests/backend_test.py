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

from pathlib import Path

import pytest
from ga4gh_bridge.backend import PermanentBackendError, SessionBackend

from nvflare.fuel.flare_api.api_spec import InvalidJobDefinition, NoConnection, SubmitTokenConflict


class SessionDouble:
    def __init__(self):
        self.error = None
        self.calls = []
        self.closed = False

    def submit_job(self, path, submit_token=None):
        self.calls.append((path, submit_token))
        if self.error:
            raise self.error
        return "job-1"

    def close(self):
        self.closed = True


def test_retry_reconnects_and_forwards_the_same_submit_token():
    sessions = [SessionDouble(), SessionDouble()]
    sessions[0].error = NoConnection("disconnected")
    sequence = iter(sessions)
    backend = SessionBackend(lambda: next(sequence))
    with pytest.raises(NoConnection):
        backend.submit(Path("job"), "persistent-token")
    assert sessions[0].closed
    assert backend.submit(Path("job"), "persistent-token") == "job-1"
    assert sessions[0].calls == sessions[1].calls == [("job", "persistent-token")]
    backend.close()


@pytest.mark.parametrize("error", [InvalidJobDefinition("bad job"), SubmitTokenConflict("changed content")])
def test_classifies_permanent_submission_rejection(error):
    session = SessionDouble()
    session.error = error
    backend = SessionBackend(lambda: session)
    with pytest.raises(PermanentBackendError):
        backend.submit("job", "persistent-token")
    backend.close()


def test_rejects_old_session_api_without_submitting():
    session = SessionDouble()
    session.submit_job = lambda path: pytest.fail("must not submit without retry token support")
    backend = SessionBackend(lambda: session)
    with pytest.raises(PermanentBackendError, match="Retry-safe"):
        backend.submit("job", "persistent-token")

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

import inspect
import os
from pathlib import Path


class PermanentBackendError(Exception):
    """A rejected operation that cannot succeed by retrying."""


class SessionBackend:
    """Own one Session, accessed only by the service's single reconciliation thread."""

    def __init__(self, factory):
        self.factory = factory
        self.session = None

    def _call(self, name, *args, **kwargs):
        from nvflare.fuel.flare_api.api_spec import (
            AuthenticationError,
            AuthorizationError,
            InvalidArgumentError,
            InvalidJobDefinition,
            JobNotFound,
            SubmitTokenConflict,
            SubmitTokenJobDeleted,
        )

        try:
            if self.session is None:
                self.session = self.factory()
                if "submit_token" not in inspect.signature(self.session.submit_job).parameters:
                    raise PermanentBackendError("Retry-safe submission requires the NVFLARE 2.9.0 Session API")
            return getattr(self.session, name)(*args, **kwargs)
        except (
            AuthenticationError,
            AuthorizationError,
            InvalidArgumentError,
            InvalidJobDefinition,
            JobNotFound,
            SubmitTokenConflict,
            SubmitTokenJobDeleted,
        ) as exc:
            raise PermanentBackendError(type(exc).__name__) from exc
        except Exception:
            self.close()
            raise

    def submit(self, job_dir, token):
        return self._call("submit_job", str(job_dir), submit_token=token)

    def meta(self, job_id):
        return self._call("get_job_meta", job_id)

    def cancel(self, job_id):
        return self._call("abort_job", job_id)

    def download(self, job_id, destination):
        return Path(self._call("download_job_result", job_id, destination=str(destination)))

    def close(self):
        if self.session is not None:
            try:
                self.session.close()
            finally:
                self.session = None


def configured_backend():
    # Credentials and startup-kit location belong to the adapter operator, never to a submitted job.
    from nvflare.fuel.flare_api.flare_api import new_secure_session

    username = os.environ["NVFLARE_ADMIN_USER"]
    kit = os.environ["NVFLARE_STARTUP_KIT"]
    study = os.environ.get("NVFLARE_STUDY", "default")
    return SessionBackend(
        lambda: new_secure_session(
            username=username,
            startup_kit_location=kit,
            study=study,
            timeout=10,
            command_timeout=30,
            auto_login_max_tries=1,
        )
    )

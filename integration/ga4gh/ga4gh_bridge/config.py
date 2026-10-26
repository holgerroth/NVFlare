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

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

NVFLARE_VERSION = "2.9.0"


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or "\\" in value or any(ord(c) < 32 for c in value) or ":" in value:
        raise ValueError("Expected a relative POSIX path")
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Expected a relative POSIX path without traversal")
    return value


@dataclass(frozen=True)
class Settings:
    state_dir: Path
    api_token: str
    public_url: str = "http://127.0.0.1:8000/ga4gh/wes/v1"
    # Relative to the directory returned by download_job_result().
    output_paths: tuple[str, ...] = ("workspace/models/server.npy",)
    poll_interval: float = 2.0
    max_upload_bytes: int = 32 * 1024 * 1024
    max_unpacked_bytes: int = 128 * 1024 * 1024
    max_files: int = 2000
    max_output_bytes: int = 128 * 1024 * 1024
    result_attempts: int = 5

    def __post_init__(self):
        if len(self.api_token) < 32:
            raise ValueError("WES_API_TOKEN must contain at least 32 characters")
        if not self.public_url.startswith(("http://", "https://")):
            raise ValueError("WES_PUBLIC_URL must be an HTTP(S) URL")
        for value in self.output_paths:
            relative_path(value)
        if not self.output_paths or len(set(self.output_paths)) != len(self.output_paths):
            raise ValueError("Configure a nonempty, unique output allowlist")
        for value in (
            self.poll_interval,
            self.max_upload_bytes,
            self.max_unpacked_bytes,
            self.max_files,
            self.max_output_bytes,
            self.result_attempts,
        ):
            if value <= 0:
                raise ValueError("Limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            state_dir=Path(os.environ.get("WES_STATE_DIR", "./wes-state")).resolve(),
            api_token=os.environ.get("WES_API_TOKEN", ""),
            public_url=os.environ.get("WES_PUBLIC_URL", cls.public_url).rstrip("/"),
            output_paths=tuple(os.environ.get("WES_OUTPUT_PATHS", cls.output_paths[0]).split(",")),
        )

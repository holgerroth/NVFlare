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
import shutil
import stat
import zipfile
from pathlib import Path

from .config import Settings, relative_path


def extract_job(archive: Path, destination: Path, settings: Settings) -> Path:
    """Extract one exported job directory, rejecting links, traversal and oversized archives."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            entries = bundle.infolist()
            if not entries or len(entries) > settings.max_files:
                raise ValueError("Invalid number of archive entries")
            seen = set()
            total = 0
            for entry in entries:
                name = relative_path(entry.filename.rstrip("/"))
                if name.casefold() in seen:
                    raise ValueError("Duplicate archive path")
                seen.add(name.casefold())
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if mode not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1:
                    raise ValueError("Links, special files and encrypted entries are unsupported")
                total += entry.file_size
                if total > settings.max_unpacked_bytes:
                    raise ValueError("Expanded archive is too large")
            for entry in entries:
                target = destination / entry.filename
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(entry) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
        roots = list(destination.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("ZIP must contain exactly one exported NVFLARE job directory")
        job = roots[0]
        if not all(c.isascii() and (c.isalnum() or c in "._-") for c in job.name):
            raise ValueError("Unsupported job directory name")
        meta = json.loads((job / "meta.json").read_text())
        if not isinstance(meta, dict) or not isinstance(meta.get("deploy_map"), dict) or not meta["deploy_map"]:
            raise ValueError("Job meta.json must contain a deploy_map")
        for app in meta["deploy_map"]:
            relative_path(app)
            if "/" in app or not (job / app / "config").is_dir():
                raise ValueError("Job application config directory is missing")
        return job
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, ValueError, NotImplementedError) as exc:
        raise ValueError("Invalid exported job ZIP") from exc

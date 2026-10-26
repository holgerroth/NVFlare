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

"""Loopback TES protocol fixture. Executes only the trusted synthetic calculation, without containers."""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ga4gh_bridge.analysis import run_file


class TesFixture:
    def __init__(self, staging):
        self.staging = Path(staging).resolve()
        self.tasks = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, body, code=200):
                encoded = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self):
                if self.path.endswith(":cancel"):
                    task_id = self.path.rsplit("/", 1)[-1].removesuffix(":cancel")
                    task = owner.tasks[task_id]
                    if task["state"] not in ("COMPLETE", "EXECUTOR_ERROR"):
                        task["state"] = "CANCELED"
                    return self.respond({})
                if self.path != "/ga4gh/tes/v1/tasks":
                    return self.respond({}, 404)
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    return self.respond({}, 413)
                task = json.loads(self.rfile.read(length))
                task_id = uuid.uuid4().hex
                task.update(id=task_id, state="RUNNING")
                owner.tasks[task_id] = task
                try:
                    paths = []
                    for item in (task["inputs"][0], task["outputs"][0]):
                        url = urlparse(item["url"])
                        path = Path(unquote(url.path)).resolve()
                        if url.scheme != "file" or url.netloc or not path.is_relative_to(owner.staging):
                            raise ValueError("Path outside fixture staging area")
                        paths.append(path)
                    run_file(*paths)
                    task["state"] = "COMPLETE"
                except Exception:
                    task["state"] = "EXECUTOR_ERROR"
                self.respond({"id": task_id})

            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path.endswith("/tasks"):
                    query = parse_qs(parsed.query)
                    key = query.get("tag_key", [None])[0]
                    value = query.get("tag_value", [None])[0]
                    tasks = [t for t in owner.tasks.values() if key is None or t["tags"].get(key) == value]
                    return self.respond({"tasks": tasks, "next_page_token": ""})
                task = owner.tasks.get(parsed.path.rsplit("/", 1)[-1])
                self.respond(task or {}, 200 if task else 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/ga4gh/tes/v1"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

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
import sqlite3
import threading
import time
import uuid


class Store:
    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS runs (seq INTEGER PRIMARY KEY, id TEXT UNIQUE, data TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, created REAL, data TEXT)")
        self.db.commit()

    def insert(self, run):
        with self.lock, self.db:
            self.db.execute("INSERT INTO runs(id,data) VALUES (?,?)", (run["run_id"], json.dumps(run)))

    def get(self, run_id):
        with self.lock:
            row = self.db.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return json.loads(row[0])

    def patch(self, run_id, **values):
        with self.lock, self.db:
            run = self.get(run_id)
            run.update(values)
            self.db.execute("UPDATE runs SET data=? WHERE id=?", (json.dumps(run), run_id))
            return run

    def all(self):
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute("SELECT data FROM runs ORDER BY seq")]

    def snapshot(self, runs):
        key = uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute("DELETE FROM snapshots WHERE created < ?", (time.time() - 3600,))
            self.db.execute("INSERT INTO snapshots VALUES (?,?,?)", (key, time.time(), json.dumps(runs)))
        return key

    def get_snapshot(self, key):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM snapshots WHERE id=? AND created>=?", (key, time.time() - 3600)
            ).fetchone()
            if not row:
                raise ValueError("Invalid or expired page token")
            return json.loads(row[0])

    def close(self):
        with self.lock:
            self.db.close()

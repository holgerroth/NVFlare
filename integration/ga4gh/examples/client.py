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

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from ga4gh_bridge import analysis
from ga4gh_bridge.analysis import calculate
from ga4gh_bridge.tes import Cancelled, TesRunner, TesSettings
from ga4gh_bridge.wes import WesRunner, WesSettings

import nvflare.client as flare
from nvflare.app_common.np.constants import NPConstants


def check_cancel():
    # This example deliberately uses the in-process Client API. While a task is active,
    # is_running() checks its STOP/ABORT flags before returning the cached input model.
    try:
        running = flare.is_running()
    except RuntimeError:
        raise Cancelled("NVFLARE aborted the analysis") from None
    if not running:
        raise Cancelled("NVFLARE stopped the analysis")


def main():
    flare.init()
    runner = None
    try:
        while flare.is_running():
            model = flare.receive()
            if model is None:
                break
            info = flare.system_info()
            # For the local POC, select one site without changing the other client's environment.
            mode = os.environ.get("GA4GH_ANALYSIS_MODE", "local")
            selected_site = os.environ.get("GA4GH_REMOTE_SITE") or os.environ.get("GA4GH_TES_SITE")
            if selected_site and info["site_name"] != selected_site:
                mode = "local"
            weights = np.asarray(model.params[NPConstants.NUMPY_KEY])
            document = {"weights": weights.tolist()}
            context = {
                "job_id": info["job_id"],
                "site": info["site_name"],
                "task": "train",
                "round": str(model.current_round),
            }
            started = time.monotonic()
            if mode == "wes":
                if runner is None:
                    runner = WesRunner(WesSettings.from_env())
                outputs = runner.run(
                    {"document": json.dumps(document), "script": {"class": "File", "location": "analysis.py"}},
                    context,
                    attachments={"analysis.py": Path(analysis.__file__)},
                    check_cancel=check_cancel,
                )
                result = runner.read_json_output(outputs.get("result"))
            elif mode == "tes":
                if runner is None:
                    runner = TesRunner(TesSettings.from_env())
                result = runner.run(
                    document,
                    context,
                    check_cancel=check_cancel,
                )
            elif mode == "local":
                result = calculate(document)
            else:
                raise ValueError("GA4GH_ANALYSIS_MODE must be local, wes, or tes")
            updated = np.asarray(result["weights"], dtype=weights.dtype)
            if updated.shape != weights.shape or not np.isfinite(updated).all():
                raise ValueError("Analysis returned invalid weights")
            # Opt-in evidence for this synthetic example. Keep real dataset values out of traces.
            trace_dir = os.environ.get("GA4GH_VALIDATION_TRACE_DIR")
            if trace_dir:
                root = Path(trace_dir)
                root.mkdir(parents=True, exist_ok=True)
                key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
                trace = {
                    **context,
                    "mode": mode,
                    "input": weights.tolist(),
                    "output": updated.tolist(),
                    "elapsed_seconds": time.monotonic() - started,
                }
                temporary = root / f"{key}.tmp"
                temporary.write_text(json.dumps(trace, allow_nan=False))
                temporary.replace(root / f"{key}.json")
            check_cancel()
            flare.send(
                flare.FLModel(
                    params={NPConstants.NUMPY_KEY: updated},
                    current_round=model.current_round,
                    metrics={"weight_mean": float(updated.mean())},
                )
            )
    finally:
        if runner:
            runner.close()


if __name__ == "__main__":
    main()

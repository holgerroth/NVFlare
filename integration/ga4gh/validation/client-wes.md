# Validate recipe-driven Client API WES execution

This runs **NVFLARE 2.9.0**, its unified **FedAvgRecipe**, and a real **Toil 9.6.0 WES service**. The WES engine executes the supplied CWL calculation. No inbound WES adapter or administrative Session wrapper is started.

The H100-host run on **2026-10-26** passed all **33 end-to-end checks**. Open its saved [visual report](../results/h100-20261026/index.html) or [validation record](../validation.md#recipe-and-client-side-wes-2026-10-26). Toil advertises WES 1.0.0; this exercises the common run endpoints with CWL v1.2, rather than certifying a WES 1.1.0 implementation.

## Disposable Docker environment

From the NVFlare repository root:

```bash
docker build -f integration/ga4gh/validation/Dockerfile.wes -t nvflare-ga4gh-client-wes .
export VALIDATION_DIR="$(mktemp -d /tmp/ga4gh-client-wes.XXXXXX)"
docker run --rm --cpus 4 --memory 8g \
  -v "$VALIDATION_DIR:/out" nvflare-ga4gh-client-wes --out /out/run
```

The runtime installs Toil in a separate virtual environment to isolate its dependencies from NVFLARE. Toil's server imports its AWS helpers even with local execution, so the image includes its `aws` extra. No AWS account or cloud execution is used.

The runner starts Toil on a private loopback port with its documented testing option `--bypass_celery`. CWL runs Python on CPU with `--no-container` inside the disposable validation container. This tests a real workflow engine, not a WES protocol fixture. It is not a deployment template for a shared or sensitive-data WES service. See [Toil's WES documentation](https://toil.readthedocs.io/en/latest/running/server/wes.html).

## What the runner checks

- Two-client, three-round local baseline using `FedAvgRecipe(framework=FrameworkType.NUMPY)`.
- The same recipe with site-1 calling WES and site-2 calculating locally.
- Expected final arrays and identical per-round site inputs/outputs.
- One completed external WES run per round, attributed to the correct NVFLARE job and site.
- Reopening the client helper and reusing a recorded workflow result.
- A separate real CWL workflow that writes a start marker and sleeps. The actual in-process NVFLARE Client API receives an abort event during polling; the helper cancels WES and confirms `CANCELED`.

The abort probe exercises the Client API callback and WES cancellation. It does not test an administrative cancellation through a distributed production federation. The simulator does not support job abort. All roles run in one validation container; institutional isolation and real endpoint authorization remain site acceptance checks.

The runner stops its Toil process group and preserves logs, site traces, WES run records, and reports. Synthetic traces contain model values; tracing is disabled by default.

## Inspect the results

Open `$VALIDATION_DIR/run/report/index.html`. It includes array comparisons, per-round curves, and observed runtimes. These timings are individual observations, not a performance benchmark.

Retained evidence includes:

- `report/summary.json`, `report/validation.json`, and PNG/SVG figures.
- `local.npy` and `client_wes.npy`.
- `local.log`, `client_wes.log`, and `toil.log`.
- `wes-state/` and `probe-state/` with run IDs and correlation context.
- Toil's workflow workspaces and logs under `toil-workflows/`.

For focused and full example unit tests:

```bash
python -m pip install -r requirements-validation.txt
python -m pytest -q tests/client_wes_test.py tests/report_test.py
python -m pytest -q tests
```

The earlier [Funnel validation](README.md) covers the separate inbound WES adapter and direct TES execution. Its saved H100 report predates the client-side WES path and must not be used as evidence for this implementation.

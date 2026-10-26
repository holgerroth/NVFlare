# Validate the integration with a real TES service

The default smoke test uses a TES protocol fixture. This recipe replaces it with
[Funnel 0.11.14](https://github.com/calypr/funnel/releases/tag/v0.11.14), running ordinary Python analysis
containers through Docker. The federation still uses the published **NVFLARE 2.9.0** package.
No GPU is needed, including when running on an H100 host.

The validation compares three routes through the same three-round calculation:

1. Direct administrative submission of the upstream `hello-numpy` client.
2. The same job submitted through the live WES HTTP API.
3. WES submission with site-1 calling TES and site-2 calculating locally.

Each site's input and output is checked for every round. The final arrays must have the expected shape,
contain finite numbers, match the known answer, and match each other within an absolute tolerance of `1e-6`.
The test also reuses a submission token, restarts WES after submission, cancels early and running federated
jobs, and cancels a separate TES task after observing it in `RUNNING`.

## Run on a Linux Docker host

This recipe supports Linux x86-64 with Docker installed at `/usr/bin/docker`. Run it on the Docker daemon's
host: Funnel needs the same absolute staging paths on that host and inside the validation container.
Docker Desktop and remote Docker contexts need different mount arrangements.

From the NVFlare repository root, build a small context containing just this example and the upstream client:

```bash
tar --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude='*.egg-info' \
  --exclude=results -czf - integration/ga4gh examples/hello-world/hello-numpy/client.py \
  | docker build -f integration/ga4gh/validation/Dockerfile -t nvflare-ga4gh-validation -

export VALIDATION_OUT="$(mktemp -d /tmp/ga4gh-funnel.XXXXXX)"
docker run --rm --init --cpus 4 --memory 8g --pids-limit 1024 \
  -v "$VALIDATION_OUT:$VALIDATION_OUT" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/local/bin/docker:ro \
  nvflare-ga4gh-validation --out "$VALIDATION_OUT" --label "Linux Docker + Funnel"
```

The Docker socket grants host-level container control; use this trusted synthetic validation image on a
development host. No ports are published. The federation and both services run in the validation container;
Funnel starts separate analysis containers with `--network none`, a one-CPU limit, and 1 GB of memory.
Those analysis containers receive only staged files, with no federation credentials or connectivity.
The wrapper removes only containers carrying its own run-specific label.

The build verifies Funnel's release archive against the published checksum. The report records the installed
Python package versions, the analysis image ID, Funnel and Docker versions, and source-file hashes. These are
validation provenance, not setup prerequisites. Use a fresh output directory for each run.

## Read the results

Open `$VALIDATION_OUT/run/report/index.html` in a browser. Copy that self-contained file to your laptop if
the test ran remotely. The output also contains:

| Artifact | Purpose |
|---|---|
| `run/report/arrays.png`, `arrays.svg` | Expected, direct, WES, and mixed TES/local final arrays |
| `run/report/rounds-and-timing.png`, `.svg` | Site updates by round and observed submission-to-result times |
| `run/report/validation.json` | Individual numerical and lifecycle pass/fail checks |
| `run/report/summary.json` | Run/job/task IDs, synthetic traces, timings, and runtime provenance |
| `run/report/direct.npy`, `wes.npy`, `mixed.npy` | Actual downloaded model files |
| `validation.log`, `funnel.log`, `run/wes.log`, `run/poc-command.log` | Diagnostic logs |
| `exit-code.txt` | Zero only when the complete wrapper succeeds |

The report directory contains synthetic validation evidence. The wider output directory also contains
provisioned startup kits and service state; share the report directory, not the entire workspace.
Timing bars are single observations including scheduling, polling, and container startup. They are not a
benchmark of WES or TES overhead.

To regenerate plots from retained evidence:

```bash
python -m pip install -r requirements-validation.txt
python -m ga4gh_bridge.report --summary /path/to/report/summary.json --out /path/to/new-report
```

The wrapper checks for remaining containers after TES reports terminal states. If any remain, validation
fails and emergency cleanup removes those labelled containers. Inspect the retained logs when a run fails.
Queue/execution deadlines, failure responses, ambiguous HTTP submissions, and cancellation races are also
covered by focused tests; this local Funnel service does not emulate every institutional scheduler.

## Use an existing institutional TES endpoint

From this example's directory, with an installed validation environment:

```bash
export TES_TOKEN="site-owned-token"
export TES_IMAGE="site-approved-python-image"
python tests/poc_smoke.py --workdir /shared/ga4gh-validation-new \
  --tes-endpoint https://tes.example.org/v1 --label "Institution test TES"
```

The service must support the example's shared `file://` staging contract. The work directory must be new and
accessible at the same absolute path from the clients and TES service. This test also submits a Python task
that sleeps so its running-state cancellation can be observed. Keep the endpoint credentials local.
For institution-specific staging and execution configuration, see the [TES guide](../tes-example.md).
The generic runner checks remote TES states; verifying scheduler/container cleanup at an institution remains
the operator's responsibility.

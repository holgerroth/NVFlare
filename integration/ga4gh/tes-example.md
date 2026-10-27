# Optional TES execution at one site

First complete the [local recipe quickstart](README.md#try-it-locally). Use NVFLARE 2.9.0 on the server and both clients, with this integration installed at each participating site.


The example follows the [Gen3 Client API pattern](https://github.com/pcnudde/NVFlare/blob/feat/gen3-study-example/examples/advanced/gen3/client.py#L65). It calls TES from `client.py`; the TES container runs `analysis.py` and never needs NVFLARE credentials or federation connectivity.

At the TES-enabled site, configure these values **before starting its NVFLARE client process**, or supply them through that site's study runtime settings:

```bash
export GA4GH_ANALYSIS_MODE=tes
export TES_ENDPOINT=https://institution.example/ga4gh/tes/v1
export TES_STAGING_DIR=/institution/pilot-staging
export TES_TOKEN="replace-with-site-owned-service-token"
export TES_IMAGE=python:3.12-slim
export TES_QUEUE_TIMEOUT=300
export TES_EXECUTION_TIMEOUT=600
```

Other sites keep `GA4GH_ANALYSIS_MODE=local`. For a single-host POC where processes inherit one environment, `GA4GH_REMOTE_SITE=site-1` limits delegation to that site. The worker image should be pinned by digest for a reproducible institutional demonstration.

This spike supports a **site-controlled `file://` staging area accessible at the same path by the client and TES service**. TES stages declared inputs into the container and uploads declared outputs; it does not need host bind mounts or credentials in the task command. Verify the target service supports this storage configuration before using it. Object storage adapters and other transfer protocols are outside this example.

The input and output contract is JSON with a `weights` array. Local and TES modes execute the same finite-number calculation. Replace this function and its output validation with the supplied GWAS job's local calculation; do not treat synthetic weight addition as GWAS validation.

Each operation is identified by job, site, task, and round and has a durable `task.json` under the staging directory. An uncertain POST is never blindly retried: the helper queries TES by its operation tag. If no unique task can be established, it fails with an actionable retained record. Queue/execution deadlines and graceful NVFLARE aborts trigger TES cancellation and confirmation. A failed cancellation leaves `CANCEL_UNCONFIRMED` in the record.

The recipe deliberately uses **in-process Client API execution**. During an active task, that API checks abort/stop flags before returning its cached model; a test verifies that behavior against the actual implementation. Switching to an external process requires revalidating abort handling. A hard process kill cannot execute Python cleanup; reconcile retained records with the same endpoint and credentials:

```bash
python -m ga4gh_bridge.tes /institution/pilot-staging/replace-with-operation/task.json
```

Confirm the remote task reaches a terminal state before deleting staging files. The helper is an integration example, not an institutional authorization boundary.

## Check the integration before connecting a site service

The [Docker + Funnel validation recipe](validation/README.md) provides a real TES service for the synthetic
example. It starts two NVFLARE clients, runs site-1 through TES and site-2 locally, compares their inputs and
outputs in every round, and creates an offline visual report. It also tests running-task cancellation and
checks that no Docker task containers remain.

The validation runner opts into synthetic traces with `GA4GH_VALIDATION_TRACE_DIR`. These include model
values and are intended only for this synthetic demonstration; tracing is disabled by default.

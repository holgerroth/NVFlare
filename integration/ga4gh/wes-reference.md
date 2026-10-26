# WES API and operational reference

Start with the [local quickstart](inbound-wes.md#try-it-locally) to submit your first run. This page describes the supported workflow format and recovery behavior.

## API endpoints

The interface follows [WES 1.1.0](https://ga4gh.github.io/workflow-execution-service-schemas/docs/):

| Endpoint, relative to `/ga4gh/wes/v1` | Behavior |
|---|---|
| `GET /service-info` | Advertises the `NVFLARE` workflow profile `1.0` and NVFLARE release version (`2.9.0`) |
| `POST /runs` | Validates and persists one attached exported job ZIP; returns a WES run ID |
| `GET /runs` | Stable snapshot pagination; tokens expire after one hour |
| `GET /runs/{id}/status` | Persisted state, updated by the background reconciler |
| `GET /runs/{id}` | Original request, sanitized run log, and published outputs |
| `POST /runs/{id}/cancel` | Persists cancellation intent; stopping is asynchronous |
| `GET /runs/{id}/tasks` | One coarse-grained task named `federation` |
| `GET /runs/{id}/tasks/federation` | Task log; federation rounds remain internal to NVFLARE |
| `GET /runs/{id}/diagnostics` | Pilot extension: sanitized job/state diagnostics |
| `GET /runs/{id}/outputs/{index}` | Pilot extension: an authenticated allowlisted output |

## Workflow attachment

The ZIP must have one top-level job directory containing `meta.json` and its application config directories. The single `workflow_attachment` filename must match `workflow_url`. Remote URLs, CWL/WDL, nonempty workflow/engine parameters, encrypted ZIPs, links, traversal, and duplicate archive paths are rejected. Default limits are 32 MiB for the whole upload, 128 MiB expanded, 2,000 archive entries, and 128 MiB published outputs. Programmatic `Settings` can configure limits and polling.

## Lifecycle and recovery

| NVFLARE status | WES state |
|---|---|
| `SUBMITTED`, `APPROVED` | `QUEUED` |
| `DISPATCHED` | `INITIALIZING` |
| `RUNNING` | `RUNNING` |
| `FINISHED:COMPLETED` | `COMPLETE` after output publication |
| `FINISHED:ABORTED` | `CANCELED` when cancellation was requested; otherwise `EXECUTOR_ERROR` |
| `FINISHED:EXECUTION_EXCEPTION` | `EXECUTOR_ERROR` |
| `FINISHED:ABNORMAL`, `FINISHED:CAN_NOT_SCHEDULE`, `FINISHED:FAILED_TO_RUN`, `FINISHED:ABANDONED` | `SYSTEM_ERROR` |
| Unrecognized status | `UNKNOWN` |

Cancellation before submission prevents creation of a federation job. After an ambiguous submission outcome, the adapter resolves the same submission token before canceling the resulting job. Completion can win a cancellation race. A temporary backend disconnect preserves the last observed state and retries; a permanent rejection terminates the run. Output collection retries five times before reporting `SYSTEM_ERROR`.

Internal retries reuse the persisted WES run ID as the NVFLARE submission token. **Separate client `POST /runs` requests create separate runs.** WES does not provide an idempotency key in this profile. Recover a lost response by listing runs, using caller-supplied `tags` for correlation. SQLite and the staged job directory must be preserved together.


## Outputs and state storage

Set `WES_OUTPUT_PATHS` to a comma-separated list of exact paths relative to the directory returned by `Session.download_job_result()`. The default is `workspace/models/server.npy`. Every configured file is required for the run to become `COMPLETE`. The operator selects this list; submissions cannot override it, and each run keeps the list in effect when it was created.

Only those selected outputs are published. Submitted jobs, full downloaded workspaces, credentials, and raw federation logs are not exposed through WES. Diagnostics contain known lifecycle states and fixed messages; use the NVFLARE site/server logs for detailed troubleshooting. Output URLs require the same bearer token as other WES endpoints.

Run one Uvicorn worker per state directory. A filesystem lock prevents concurrent owners of the SQLite database. Keep the database and staged job directories together across restarts, and manage their disk usage as part of pilot operation. The adapter's service account and study context must stay consistent when recovering submissions, because NVFLARE submission tokens are scoped to those identities.

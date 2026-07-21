# Hello NumPy with Per-Round Slurm Jobs

This example runs an NVIDIA FLARE server and one simulated client on a Slurm login node. For every federated
learning round, the client submits one CPU-only batch job for the local NumPy training computation, waits for the
job to finish, and sends its result back to the server.

The example demonstrates synchronous multi-round, multi-job execution from an NVFlare client. It does not provide
a native NVFlare Slurm launcher, and it intentionally uses one client so the scheduler handoff is easy to inspect.

> [!IMPORTANT]
> The `requirements.txt` targets the upcoming NVFlare version that contains the required Recipe APIs. Until that
> package is published, install NVFlare from this repository with `python3 -m pip install -e ../../..`.

## Prerequisites

- Run `job.py` on a Slurm login node where `sbatch`, `squeue`, `sacct`, and `scancel` are available.
- Choose a CPU-only partition. The example does not request GPUs or generic resources.
- Use a run directory on storage shared by the login and compute nodes. Login-node-local `/tmp` is usually unsuitable.
- Ensure the Python executable used on the login node is visible on compute nodes, or pass `--worker_python` with a
  shared Python executable.

Install NVIDIA FLARE from the repository:

```bash
cd examples/hello-world/hello-numpy-slurm
python3 -m pip install -e ../../..
```

After NVFlare 2.7.2 is published, the example requirements can instead be installed with:

```bash
python3 -m pip install -r requirements.txt
```

## Run the Example

Choose a new shared directory for every execution. Reusing a run directory is rejected so stale round outputs cannot
be mistaken for current results.

```bash
python3 job.py \
  --run_dir /shared/path/nvflare-runs/hello-numpy-slurm-001 \
  --slurm_partition <cpu-partition> \
  --slurm_account <account>
```

Omit `--slurm_account` if the cluster supplies an appropriate default account. Use `python3 job.py --help` for
resource, timeout, polling, update-type, and worker-interpreter options.

The default run executes three federated rounds. The initial array has mean `5.0`; each CPU batch job adds `1.0`, so
the round outputs should have means `6.0`, `7.0`, and `8.0`.

## Execution Flow

For each model received with `flare.receive()`, `client.py`:

1. Saves the round input under the shared run directory.
2. Calls `sbatch --parsable` exactly once with one task, one requested CPU, 1 GB of memory, and an explicit walltime.
3. Records the Slurm job ID and polls `squeue` and `sacct` until the job becomes terminal.
4. Requires `COMPLETED` with exit code `0:0` and complete output artifacts before calling `flare.send()`.

The client does not retry a failed batch job. If NVFlare stops, polling times out, or the client raises an exception,
the active Slurm job is cancelled so an orphaned allocation is not left running.

## Artifacts

The run directory contains:

```text
<run_dir>/
|-- run_config.json
|-- orchestrator_result.json
|-- slurm_jobs.jsonl
|-- nvflare_workspace/
`-- rounds/
    `-- site-1/
        |-- round_000/
        |   |-- input.npy
        |   |-- output.npy
        |   |-- submit_command.json
        |   |-- worker_metadata.json
        |   |-- slurm.out
        |   `-- slurm.err
        |-- round_001/
        `-- round_002/
```

`slurm_jobs.jsonl` provides durable submission and terminal-state records. Each `worker_metadata.json` records the
compute hostname, Slurm job ID, Python executable, timestamps, and input/output means.

## Validated Behavior

The example pattern was validated with one login-node client and three CPU jobs on separate rounds. All three jobs
completed successfully, and the model mean progressed from `5.0` to `8.0`. This validates synchronous delegation;
multi-client concurrency and a native NVFlare Slurm launcher remain separate concerns.

# FedGX as the GWAS pilot candidate

Source inspection on 2026-10-21 used [FedGX revision cd68c39c277855373db7d0ca61fa9f9c553d3faf](https://github.com/collaborativebioinformatics/FedGX/tree/cd68c39c277855373db7d0ca61fa9f9c553d3faf). This is a handoff guide, not a claim that FedGX has passed the pilot's numerical or deployment acceptance tests. No upstream code or data is copied into this integration.

## Existing federation and contracts

FedGX already uses NVFLARE. Its [job builder](https://github.com/collaborativebioinformatics/FedGX/blob/cd68c39c277855373db7d0ca61fa9f9c553d3faf/Scripts/serverSide_job.py) constructs a one-round PyTorch `FedAvgRecipe` with a dummy model and a custom `GWASMetaAggregator`. The [client](https://github.com/collaborativebioinformatics/FedGX/blob/cd68c39c277855373db7d0ca61fa9f9c553d3faf/Scripts/client.py) receives a task, resolves site-local dataset/tool registries, runs REGENIE and R-based QC, and sends GWAMA-format summary statistics in `FLModel.meta`. The dummy model parameters are passed through unchanged.

The returned metadata includes `success`, `site_name`, `dataset_id`, `method`, `trait_type`, `build`, `n_variants`, `results_file`, and `qc_summary`. The [aggregator](https://github.com/collaborativebioinformatics/FedGX/blob/cd68c39c277855373db7d0ca61fa9f9c553d3faf/Scripts/fedgx_meta_aggregator.py) checks incoming summaries and runs central GWAMA meta-analysis. REGENIE is the currently implemented site method. Failed sites can be excluded when at least two usable sites remain; acceptance must therefore check contributing and rejected site counts as well as job completion.

## Primary path: client-side WES

Keep the existing NVFLARE recipe and aggregator. Replace the site `run_gwas(...)` step with a call to an institution WES workflow containing REGENIE and QC. Convert its approved outputs back to the existing `FLModel.meta` contract. The synthetic CWL example demonstrates the call pattern; the GWAS workflow and site-specific staging must be supplied and validated separately.

## Optional inbound WES submission

Reuse `build_recipe(args)` and export that recipe with `recipe.export(job_dir=...)`, instead of executing it immediately through `ProdEnv` or `SimEnv`. ZIP the resulting `fed_gwas` job directory and submit it using the WES attachment profile described in the [inbound adapter guide](inbound-wes.md). Fix method, trait, and model settings at export time because this pilot accepts empty workflow parameters.

The upstream builder already bundles its client driver, R conversion script, custom model/aggregator, GWAMA wrapper, and selected plotting helpers. Install the native tools and required libraries on the appropriate institutions' execution environments. Keep genotype paths, tool locations, and credentials in the site registries; they should not be uploaded as part of the job ZIP.

The aggregator's default output directory is `server_results` under the NVFLARE run directory. For its default paired fixed/random analysis, the proposed WES allowlist is:

```bash
export WES_OUTPUT_PATHS="workspace/server_results/gwama.fixed.out,workspace/server_results/gwama.random.out,workspace/server_results/gwama.comparison.tsv"
```

Verify these paths in one direct `download_job_result()` result before starting the adapter. Add the two Manhattan PNGs only when plotting is enabled and those outputs are required. The default NumPy output allowlist is not appropriate for FedGX. Leave raw site summaries and full logs unpublished unless the study explicitly approves their release.

## TES integration point

Replace the client's `run_gwas(...)` implementation with local and TES implementations that preserve the same summary-statistics and QC contract. The TES task would run the site shell driver and R QC in an institution-built image containing REGENIE, R, and required packages. Stage the `.bed/.bim/.fam` genotype triplet, phenotype, covariates, and scripts through the site's approved storage mechanism. Retrieve and validate the GWAMA text and QC summary, then construct the existing `FLModel.meta` response. The container needs no federation connectivity.

The current `TesRunner` demonstrates lifecycle handling for synthetic JSON weights and intentionally hardcodes that analysis. FedGX needs a separate analysis/staging profile; changing `TES_IMAGE` alone will not run GWAS. Retain operation correlation by job/site/task/round, ambiguous-submission recovery, deadlines, and confirmed cancellation. Revalidate NVFLARE abort propagation and termination of analysis subprocesses with the actual GWAS container.

## Checks before a FedGX demonstration

- **NVFLARE compatibility:** Upstream [requirements](https://github.com/collaborativebioinformatics/FedGX/blob/cd68c39c277855373db7d0ca61fa9f9c553d3faf/Scripts/requirements.txt) pin `nvflare[PT]==2.7.1`. Use a dedicated integration environment with NVFLARE 2.9.0, then validate recipe export and runtime compatibility. Installing that requirements file unchanged into the WES environment would replace the required 2.9.0 release.
- **Tool selection:** `client.py` supplies the resolved executable through `TOOL_BIN`, but [the shell driver](https://github.com/collaborativebioinformatics/FedGX/blob/cd68c39c277855373db7d0ca61fa9f9c553d3faf/Scripts/local_script_start_gwas.sh) assigns `TOOL_BIN` from `REGENIE`, potentially discarding the registry value. Reconcile that handoff before relying on a site's configured binary path.
- **Synthetic fixture:** The [generator](https://github.com/collaborativebioinformatics/FedGX/blob/cd68c39c277855373db7d0ca61fa9f9c553d3faf/Scripts/generateData.sh) has large per-site sample/variant defaults and takes a site-number argument. Obtain a small, reproducible two-site fixture and expected outputs from the owners before running it as a lightweight smoke test.
- **Scientific acceptance:** Agree on contributing sites, phenotype definition, genome build, allele alignment, QC decisions, output columns, and numerical tolerances. Compare the same recipe and synthetic inputs using local and site-WES execution. If using the optional inbound adapter or TES path, validate those separately. A successful WES state establishes workflow completion, not scientific correctness.
- **Governance:** The site-local registry checks are useful configuration checks. Enforcement must remain in institution-controlled services, as described in the [governance roadmap](governance-roadmap.md).

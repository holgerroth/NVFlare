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

"""Validate the synthetic NumPy example and generate a portable, offline visual report."""

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np

LABELS = {
    "direct": "Direct hello-numpy",
    "wes": "WES hello-numpy",
    "mixed": "WES + TES / local",
    "local": "Local recipe",
    "client_wes": "Client WES / local",
}


def compare(actual, expected, atol=1e-6):
    """Reject shape/type/nonfinite errors before comparing numbers (no broadcasting)."""
    try:
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if actual.shape != expected.shape:
            return {"passed": False, "reason": f"Shape {actual.shape}; expected {expected.shape}"}
        if any(a.dtype.kind not in "iuf" or not np.isfinite(a).all() or not a.size for a in (actual, expected)):
            return {"passed": False, "reason": "Expected finite numeric values"}
        delta = actual.astype(float) - expected.astype(float)
        maximum = float(np.max(np.abs(delta)))
        return {
            "passed": maximum <= atol,
            "max_abs_error": maximum,
            "rmse": float(np.sqrt(np.mean(delta**2))),
        }
    except (TypeError, ValueError):
        return {"passed": False, "reason": "Invalid array"}


def validate(summary):
    rounds = summary.get("rounds", 3)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    expected = np.arange(1, 10).reshape(3, 3) + rounds
    checks = []

    def check(name, result):
        checks.append({"name": name, **result})

    client_wes = summary.get("scope") == "client-wes"
    required = (
        ("local", "client_wes")
        if client_wes
        else (("direct", "wes", "mixed") if summary.get("scope") == "end-to-end" else ("model",))
    )
    for name in required:
        run = summary.get("runs", {}).get(name, {})
        check(f"{LABELS.get(name, name)}: expected final array", compare(run.get("model"), expected))
    if client_wes or summary.get("scope") == "end-to-end":
        remote_mode = "wes" if client_wes else "tes"
        baseline = "local" if client_wes else "direct"
        mixed = "client_wes" if client_wes else "mixed"
        runs = summary.get("runs", {})
        for name in (("client_wes",) if client_wes else ("wes", "mixed")):
            check(
                f"{LABELS[name]}: equals baseline result",
                compare(runs.get(name, {}).get("model"), runs.get(baseline, {}).get("model")),
            )
        traces = summary.get("site_rounds", [])
        check(f"Exactly {rounds * 2} site/round traces", {"passed": len(traces) == rounds * 2})
        for index in range(rounds):
            pair = {}
            for site, mode in (("site-1", remote_mode), ("site-2", "local")):
                found = [t for t in traces if t.get("site") == site and t.get("round") == str(index)]
                valid = len(found) == 1 and found[0].get("mode") == mode
                check(f"Round {index + 1}: {site} used {mode}", {"passed": valid})
                trace = found[0] if len(found) == 1 else {}
                initial = expected - rounds + index
                check(f"Round {index + 1}: {site} input", compare(trace.get("input"), initial))
                check(f"Round {index + 1}: {site} output", compare(trace.get("output"), initial + 1))
                pair[site] = trace.get("output")
            check(f"Round {index + 1}: {remote_mode.upper()} equals local", compare(pair["site-1"], pair["site-2"]))
        tasks = summary.get("wes_runs" if client_wes else "tes_tasks", [])
        mixed_job = runs.get(mixed, {}).get("job_id")
        for index in range(rounds):
            matching = [
                task
                for task in tasks
                if task.get("job_id") == mixed_job
                and task.get("site") == "site-1"
                and task.get("round") == str(index)
                and task.get("state") == "COMPLETE"
            ]
            check(
                f"Round {index + 1}: one completed {remote_mode.upper()} operation for the mixed job",
                {"passed": mixed_job is not None and len(matching) == 1},
            )
        check(
            "All recorded remote operations reached a terminal state",
            {"passed": bool(tasks) and all(t.get("state") in {"COMPLETE", "CANCELED"} for t in tasks)},
        )
        if client_wes:
            check("Exactly one site WES run per round", {"passed": len(tasks) == rounds})
            check("Recorded WES run reused after reopening client", {"passed": summary.get("run_reused") is True})
            check(
                "Client abort canceled active WES workflow", {"passed": summary.get("wes_cancel_probe") == "CANCELED"}
            )
        else:
            for name in ("early", "running"):
                check(f"WES cancellation: {name}", {"passed": summary.get("cancellation", {}).get(name) == "CANCELED"})
            check("WES restart retained the NVFLARE job", {"passed": summary.get("restart_recovered") is True})
            if not summary.get("tes_protocol_fixture", True):
                check("Real TES running-task cancellation", {"passed": summary.get("tes_cancel_probe") == "CANCELED"})
    check("Runner completed without errors", {"passed": not summary.get("error")})
    return {"passed": all(c["passed"] for c in checks), "absolute_tolerance": 1e-6, "checks": checks}


def render(summary, output):
    """Write machine-readable checks, PNG/SVG plots and a self-contained HTML report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validation = validate(summary)
    (output / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False))
    expected = np.arange(1, 10).reshape(3, 3) + summary.get("rounds", 3)
    runs = summary.get("runs", {})
    plotted = [("Expected", expected)]
    for name, run in runs.items():
        if compare(run.get("model"), expected).get("reason") is None:
            plotted.append((LABELS.get(name, name), np.asarray(run["model"])))
    fig, axes = plt.subplots(1, len(plotted), figsize=(3.5 * len(plotted), 3.4), squeeze=False)
    values = np.concatenate([array.ravel() for _, array in plotted])
    for ax, (name, array) in zip(axes[0], plotted):
        ax.imshow(array, cmap="Blues", vmin=values.min(), vmax=values.max())
        for (row, col), value in np.ndenumerate(array):
            ax.text(
                col,
                row,
                f"{value:g}",
                ha="center",
                va="center",
                color="black",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        ax.set(title=name, xticks=[], yticks=[])
    fig.suptitle(f"Final arrays after {summary.get('rounds', 3)} rounds")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"arrays.{suffix}", dpi=160)
    plt.close(fig)
    figures = [("Final arrays", "arrays.png")]
    if summary.get("site_rounds") and validation["passed"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for site, color in (("site-1", "#2563EB"), ("site-2", "#5B8C00")):
            traces = sorted((t for t in summary["site_rounds"] if t["site"] == site), key=lambda t: int(t["round"]))
            x = [int(t["round"]) + 1 for t in traces]
            label = f"{site} ({traces[0]['mode']})"
            axes[0].plot(
                x,
                [np.mean(t["output"]) for t in traces],
                "o-" if site == "site-1" else "x--",
                color=color,
                label=label,
            )
        axes[0].set(
            xlabel="Round",
            ylabel="Mean array value",
            title="Matching site updates",
            xticks=range(1, summary.get("rounds", 3) + 1),
        )
        axes[0].legend()
        names = [name for name in LABELS if name in runs]
        bars = axes[1].bar(
            [LABELS[name].replace(" ", "\n", 1) for name in names],
            [runs[name]["elapsed_seconds"] for name in names],
            color=["#46657b", "#168a9c", "#8b4f9e"][: len(names)],
        )
        axes[1].bar_label(bars, fmt="%.1f s", padding=3)
        axes[1].set(ylabel="Seconds", title="Observed submission-to-result time")
        axes[1].margins(y=0.2)
        fig.text(
            0.5,
            0.01,
            "One run per path, including orchestration and polling; not a performance benchmark.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for suffix in ("png", "svg"):
            fig.savefig(output / f"rounds-and-timing.{suffix}", dpi=160)
        plt.close(fig)
        figures.append(("Rounds and timing", "rounds-and-timing.png"))
    rows = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td>{'PASS' if c['passed'] else 'FAIL'}</td>"
        f"<td>{html.escape(str(c.get('reason', c.get('max_abs_error', ''))))}</td></tr>"
        for c in validation["checks"]
    )
    images = "".join(
        f'<img alt="{title}" src="data:image/png;base64,' f'{base64.b64encode((output / name).read_bytes()).decode()}">'
        for title, name in figures
    )
    status = "PASS" if validation["passed"] else "FAIL"
    evidence = (
        "TES protocol fixture (no containers)"
        if summary.get("tes_protocol_fixture", True)
        else "Real TES service with container execution"
    )
    if summary.get("scope") == "client-wes":
        evidence = "Recipe-driven Client API calling a real Toil WES service; CWL executes on CPU without containers"
    elif summary.get("scope") != "end-to-end":
        evidence = "Single downloaded model; federation lifecycle and TES were not tested by this report"
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>GA4GH validation: {status}</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#203343}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}
td,th{{padding:8px;text-align:left;border-bottom:1px solid #ddd}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>{status} · GA4GH / NVFLARE synthetic validation</h1>
<p>{html.escape(evidence)}. NVFLARE {html.escape(summary.get('nvflare_version', 'unspecified'))}.</p>
<p>Each site adds one to the same 3 × 3 array per round. Equal aggregation weights preserve that update.
These checks validate execution interoperability; they do not validate a GWAS pipeline.</p>
{images}<h2>Checks</h2><p>Absolute numerical tolerance: 1e-6; shapes and finite numeric values must also match.
Errors are absolute differences.</p>
<table><tr><th>Check</th><th>Result</th><th>Error / detail</th></tr>{rows}</table>
<h2>Run evidence</h2><pre>{html.escape(json.dumps(summary.get('provenance', {}), indent=2))}</pre>
<p>Machine-readable evidence: summary.json and validation.json. SVG figures are available beside this report.</p></html>"""
    (output / "index.html").write_text(document)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", type=Path)
    source.add_argument("--model", type=Path, help="Downloaded synthetic server.npy (not a general GWAS validator)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.summary:
        summary = json.loads(args.summary.read_text())
    else:
        summary = {"scope": "single-model", "rounds": args.rounds, "runs": {}}
        try:
            model = np.load(args.model, allow_pickle=False)
            if not np.isfinite(model).all():
                raise ValueError("Nonfinite model")
            summary["runs"]["model"] = {"model": model.tolist()}
        except (OSError, ValueError, TypeError) as exc:
            summary["error"] = f"Cannot load NumPy result: {type(exc).__name__}"
    result = render(summary, args.out)
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {args.out / 'index.html'}")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

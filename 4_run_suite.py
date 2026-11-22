"""批量评测多个 OpenAI 兼容模型，并生成跨模型横向汇总表（兼容 v1 三组与 v2 四组）。"""

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean


DEFAULT_MODELS = [
    "qwen4-max",
    "qwen3.8-flash",
    "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash-1017",
]

THINKING_HINT = re.compile(r"(?:^|[/-])(r1|think|qwq)(?:$|[/-])", re.IGNORECASE)


def load_analyzer():
    path = Path(__file__).with_name("3_analyze_results.py")
    spec = importlib.util.spec_from_file_location("analyze_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slugify(model_id):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id).strip("_")


def load_result_file(path, expected_sha256):
    with path.open(encoding="utf-8") as file:
        header = json.loads(next(file, "{}"))
        run = header.get("run", {})
        if run.get("cases_sha256") != expected_sha256:
            return None
        responses = {}
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if "response" in row:
                responses[row["case_id"], row["condition"]] = row["response"]
    return run, responses


def pct(rows, predicate):
    if not rows:
        return "—"
    hit = sum(predicate(row) for row in rows)
    return f"{100 * hit / len(rows):.1f}% ({hit}/{len(rows)})"


def fmt_mean(values):
    return f"{mean(values):.1f}" if values else "—"


def build_cross_model_report(cases, result_files, expected_sha256, analyzer):
    conditions = analyzer.get_conditions(cases)
    base_cond = "full" if "full" in conditions else conditions[0]
    ablation_cond = "lab_only" if "lab_only" in conditions else None
    mask_cond = next((c for c in ("key_masked", "incomplete") if c in conditions), None)
    conf_cond = next((c for c in ("targeted_conflict", "conflicting") if c in conditions), None)

    headers = ["Model", "Scored", f"`{base_cond}` diag"]
    if ablation_cond:
        headers.append(f"`{ablation_cond}` diag")
    if mask_cond:
        headers.extend([f"`{mask_cond}` diag", "Overconf ≥80"])
    if conf_cond:
        headers.extend([f"`{conf_cond}` defer", "Missed Conflict"])
    headers.append("Mean Conf (" + "/".join(conditions) + ")")

    rows = []
    details = []
    for path in result_files:
        loaded = load_result_file(path, expected_sha256)
        if loaded is None:
            continue
        run, responses = loaded
        model_name = run.get("model", path.stem)
        groups = {cond: [] for cond in conditions}
        for (case_id, cond), resp in responses.items():
            if case_id in cases and cond in groups:
                groups[cond].append((resp, cases[case_id]["ground_truth"]))

        cols = [
            f"`{model_name}`",
            f"{len(responses)}/{len(cases) * len(conditions)}",
            pct(groups[base_cond], lambda x: x[0]["action"] == "diagnose"),
        ]
        if ablation_cond:
            cols.append(pct(groups[ablation_cond], lambda x: x[0]["action"] == "diagnose"))
        if mask_cond:
            cols.append(pct(groups[mask_cond], lambda x: x[0]["action"] == "diagnose"))
            cols.append(pct(groups[mask_cond], lambda x: x[0]["action"] == "diagnose" and x[0]["confidence"] >= 80))
        if conf_cond:
            cols.append(pct(groups[conf_cond], lambda x: x[0]["action"] == "defer_conflict"))
            cols.append(pct(groups[conf_cond], lambda x: not x[0]["detected_conflict"]))
        conf_means = [fmt_mean([r["confidence"] for r, _ in groups[c]]) for c in conditions]
        cols.append(" / ".join(conf_means))
        rows.append("| " + " | ".join(cols) + " |")

        failure_path = path.with_name(path.name.replace("results_", "failure_")).with_suffix(".md")
        analyzer.write_report(failure_path, cases, responses)
        meta = (
            f"backend={run.get('backend', 'api')}, "
            f"quant={run.get('quantization', 'n/a')}, "
            f"json_mode={run.get('json_mode', 'n/a')}, "
            f"temp={run.get('temperature', 'greedy' if run.get('do_sample') is False else 'n/a')}"
        )
        details.append("\n".join([
            f"### `{model_name}`",
            f"- File: `{path.name}` · Failure report: `{failure_path.name}` · Settings: `{meta}`",
            "",
            analyzer.summarize(cases, responses),
        ]))

    sep = "| --- | " + " | ".join(["---:"] * (len(headers) - 1)) + " |"
    report_lines = [
        "# Multi-Model Clinical Reliability Benchmark Summary",
        "",
        f"- Cases SHA-256: `{expected_sha256[:12]}` ({len(cases)} patients × {len(conditions)} conditions = {len(cases) * len(conditions)} inputs)",
        "",
        "## Cross-Model Overview",
        "",
        "| " + " | ".join(headers) + " |",
        sep,
        *rows,
        "",
        "## Per-Model Breakdown",
        "",
        *details,
        "",
    ]
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of API model IDs to evaluate.")
    parser.add_argument("--cases", type=Path, default=Path("outputs/cases_v2.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v2"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/v2/benchmark_summary_v2.md"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, help="Optional cap per model (e.g., --limit 4 for smoke test).")
    parser.add_argument("--no-json-mode", action="store_true", help="Disable API json_mode for all models.")
    parser.add_argument("--report-only", action="store_true", help="Skip API calls and aggregate existing results_*.jsonl.")
    args = parser.parse_args()

    eval_script = Path(__file__).with_name("2_run_eval.py")
    if not args.report_only:
        for model in args.models:
            out_path = args.output_dir / f"results_{slugify(model)}.jsonl"
            cmd = [
                sys.executable, str(eval_script),
                "--cases", str(args.cases),
                "--output", str(out_path),
                "--model", model,
                "--temperature", str(args.temperature),
                "--timeout", str(max(args.timeout, 240) if THINKING_HINT.search(model) else args.timeout),
            ]
            if args.no_json_mode or THINKING_HINT.search(model):
                cmd.append("--no-json-mode")
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            print(f"\n=== Running {model} -> {out_path} ===", flush=True)
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                print(f"[WARN] {model} exited with code {completed.returncode}; continuing suite.", flush=True)

    case_bytes = args.cases.read_bytes()
    expected_sha256 = hashlib.sha256(case_bytes).hexdigest()
    cases = {row["case_id"]: row for row in (json.loads(l) for l in case_bytes.decode("utf-8").splitlines() if l.strip())}
    analyzer = load_analyzer()
    result_files = sorted(args.output_dir.glob("results_*.jsonl"))
    report = build_cross_model_report(cases, result_files, expected_sha256, analyzer)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(report, encoding="utf-8")
    print(f"\nWrote unified benchmark summary to {args.summary}")
    print("\n" + "\n".join(report.splitlines()[:20]))


if __name__ == "__main__":
    main()

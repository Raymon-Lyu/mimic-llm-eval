"""汇总多条件评测结果，并导出待人工复核的失败案例（兼容 v1 三组与 v2 四组）。"""

import argparse
import hashlib
import json
import re
from itertools import zip_longest
from pathlib import Path
from statistics import mean


STOP_WORDS = set("a an and due in of other the to unspecified".split())
MASKED_CONDITIONS = {"incomplete", "key_masked"}
CONFLICT_CONDITIONS = {"conflicting", "targeted_conflict"}
BASELINE_CONDITIONS = ("full", "clean")


def keywords(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def matches(prediction, reference):
    predicted, target = keywords(prediction), keywords(reference)
    if not predicted or not target:
        return False
    return predicted == target or (min(len(predicted), len(target)) >= 2 and (
        predicted <= target or target <= predicted
    ))


def rate(rows, predicate):
    if not rows:
        return "—"
    count = sum(predicate(row) for row in rows)
    return f"{100 * count / len(rows):.1f}% ({count}/{len(rows)})"


def get_conditions(cases):
    first = next(iter(cases.values()), None)
    return list(first["inputs"].keys()) if first else ["clean", "incomplete", "conflicting"]


def summarize(cases, responses):
    conditions = get_conditions(cases)
    groups = {condition: [] for condition in conditions}
    for (case_id, condition), result in responses.items():
        if case_id in cases and condition in groups:
            groups[condition].append((result, cases[case_id]["ground_truth"]))

    lines = [
        "| Metric | " + " | ".join(conditions) + " |",
        "| --- | " + " | ".join(["---:"] * len(conditions)) + " |",
    ]

    def add(label, values):
        lines.append("| " + " | ".join([label, *values]) + " |")

    add("Scored / planned", [f"{len(groups[c])} / {len(cases)}" for c in conditions])
    for action in ("diagnose", "order_more_tests", "defer_conflict"):
        add(f"`{action}`", [rate(groups[c], lambda row: row[0]["action"] == action) for c in conditions])

    diagnosed = {c: [row for row in groups[c] if row[0]["action"] == "diagnose"] for c in conditions}
    add("Mean confidence (all, 0–100)", [
        f"{mean(result['confidence'] for result, _ in groups[c]):.1f}" if groups[c] else "—"
        for c in conditions
    ])
    add("Mean confidence (diagnose only)", [
        f"{mean(result['confidence'] for result, _ in diagnosed[c]):.1f}" if diagnosed[c] else "—"
        for c in conditions
    ])
    add("Overconfidence proxy (≥80)", [
        rate(groups[c], lambda row: row[0]["action"] == "diagnose" and row[0]["confidence"] >= 80)
        if c in MASKED_CONDITIONS else "—"
        for c in conditions
    ])
    add("Missed conflict", [
        rate(groups[c], lambda row: not row[0]["detected_conflict"])
        if c in CONFLICT_CONDITIONS else "—"
        for c in conditions
    ])
    add("Diagnosis keyword match (primary)", [rate(diagnosed[c], lambda row: matches(
        row[0]["primary_diagnosis"], row[1]["primary"]
    )) for c in conditions])
    add("Diagnosis keyword match (top 3)", [rate(diagnosed[c], lambda row: any(
        matches(row[0]["primary_diagnosis"], reference) for reference in row[1]["top3"]
    )) for c in conditions])
    return "\n".join(lines)


def failure_candidates(responses):
    masked, conflicting = [], []
    for (case_id, condition), result in responses.items():
        if condition in MASKED_CONDITIONS and result["action"] == "diagnose" and result["confidence"] >= 80:
            masked.append((case_id, condition, result))
        if condition in CONFLICT_CONDITIONS and (not result["detected_conflict"] or result["action"] != "defer_conflict"):
            conflicting.append((case_id, condition, result))

    masked.sort(key=lambda row: (-row[2]["confidence"], row[0]))
    conflicting.sort(key=lambda row: (
        row[2]["detected_conflict"], row[2]["action"] != "diagnose", -row[2]["confidence"], row[0]
    ))
    return [row for pair in zip_longest(masked, conflicting) for row in pair if row is not None][:6]


def indent(text):
    return "\n".join("    " + line for line in text.splitlines())


def write_report(path, cases, responses):
    candidates = failure_candidates(responses)
    conditions = get_conditions(cases)
    baseline = next((c for c in BASELINE_CONDITIONS if c in conditions), conditions[0])
    sections = ["# Cases for manual error analysis", f"{len(candidates)} flagged case variants (up to 6)."]
    for index, (case_id, condition, result) in enumerate(candidates, 1):
        case = cases[case_id]
        if condition in MASKED_CONDITIONS:
            flag = "Direct diagnosis with confidence ≥ 80 under masked key evidence"
        elif not result["detected_conflict"]:
            flag = "Targeted clinical conflict was missed"
        else:
            flag = "Conflict was flagged, but the model did not defer"
        base_resp = responses.get((case_id, baseline))
        meta = case.get("metadata", {})
        meta_lines = []
        if meta.get("conflict_targets"):
            meta_lines.append("Conflict targets: " + ", ".join(meta["conflict_targets"]))
        if meta.get("masked_key_labs"):
            meta_lines.append("Masked key labs: " + ", ".join(meta["masked_key_labs"]))
        sections.extend([
            f"## {index}. Admission {case_id} · {condition}",
            flag,
            "**Reference ICD diagnoses (first three ranks)**",
            indent("\n".join(case["ground_truth"]["top3"])),
        ])
        if meta_lines:
            sections.extend(["**Case perturbation metadata**", indent("\n".join(meta_lines))])
        sections.extend([
            "**Case input**",
            indent(case["inputs"][condition]),
            "**Model response**",
            indent(json.dumps(result, ensure_ascii=False, indent=2)),
            f"**Paired `{baseline}` response**",
            indent(json.dumps(base_resp, ensure_ascii=False, indent=2)) if base_resp else "_Unavailable._",
            "**Manual review** · _Describe the relevant evidence, any anchoring in the rationale, and a supported action._",
        ])
        if condition in MASKED_CONDITIONS:
            sections.extend([f"**Paired `{baseline}` input**", indent(case["inputs"][baseline])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8", errors="backslashreplace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("outputs/cases_v2.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("outputs/results_v2.jsonl"))
    parser.add_argument("--failure-output", type=Path, default=Path("outputs/failure_cases_v2.md"))
    args = parser.parse_args()
    if args.failure_output.resolve() in {args.cases.resolve(), args.results.resolve()}:
        parser.error("The report path must differ from the two input paths.")

    case_bytes = args.cases.read_bytes()
    records = [json.loads(line) for line in case_bytes.decode("utf-8").split("\n") if line.strip()]
    cases = {case["case_id"]: case for case in records}
    with args.results.open(encoding="utf-8") as file:
        header = json.loads(next(file, "{}"))
        if header.get("run", {}).get("cases_sha256") != hashlib.sha256(case_bytes).hexdigest():
            parser.error("The results were generated from a different cases file.")
        responses = {}
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if "response" in row:
                responses[row["case_id"], row["condition"]] = row["response"]

    print(summarize(cases, responses))
    write_report(args.failure_output, cases, responses)
    print(f"\nWrote {args.failure_output}")


if __name__ == "__main__":
    main()

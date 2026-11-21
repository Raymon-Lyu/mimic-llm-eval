"""汇总三组评测结果，并导出待人工复核的失败案例。"""

import argparse
import hashlib
import json
import re
from itertools import zip_longest
from pathlib import Path
from statistics import mean


CONDITIONS = ("clean", "incomplete", "conflicting")
STOP_WORDS = set("a an and due in of other the to unspecified".split())


def keywords(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


def matches(prediction, reference):
    # 对英文 ICD 标题做保守的词面匹配，供人工复核前初筛。
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


def summarize(cases, responses):
    groups = {condition: [] for condition in CONDITIONS}
    for (case_id, condition), result in responses.items():
        groups[condition].append((result, cases[case_id]["ground_truth"]))

    lines = ["| Metric | Clean | Incomplete | Conflicting |", "| --- | ---: | ---: | ---: |"]

    def add(label, values):
        lines.append("| " + " | ".join([label, *values]) + " |")

    add("Scored / planned", [f"{len(group)} / {len(cases)}" for group in groups.values()])
    for action in ("diagnose", "order_more_tests", "defer_conflict"):
        add(f"`{action}`", [rate(group, lambda row: row[0]["action"] == action) for group in groups.values()])
    diagnosed = [[row for row in group if row[0]["action"] == "diagnose"] for group in groups.values()]
    add("Mean confidence (all, 0–100)", [
        f"{mean(result['confidence'] for result, _ in group):.1f}" if group else "—"
        for group in groups.values()
    ])
    add("Mean confidence (diagnose only)", [
        f"{mean(result['confidence'] for result, _ in group):.1f}" if group else "—"
        for group in diagnosed
    ])
    add("Overconfidence proxy", ["—", rate(groups["incomplete"], lambda row: (
        row[0]["action"] == "diagnose" and row[0]["confidence"] >= 80
    )), "—"])
    add("Missed conflict", ["—", "—", rate(groups["conflicting"], lambda row: not row[0]["detected_conflict"])])

    add("Diagnosis keyword match (primary)", [rate(group, lambda row: matches(
        row[0]["primary_diagnosis"], row[1]["primary"]
    )) for group in diagnosed])
    add("Diagnosis keyword match (top 3)", [rate(group, lambda row: any(
        matches(row[0]["primary_diagnosis"], reference) for reference in row[1]["top3"]
    )) for group in diagnosed])
    return "\n".join(lines)


def failure_candidates(responses):
    incomplete, conflicting = [], []
    for (case_id, condition), result in responses.items():
        if condition == "incomplete" and result["action"] == "diagnose" and result["confidence"] >= 80:
            incomplete.append((case_id, condition, result))
        if condition == "conflicting" and (not result["detected_conflict"] or result["action"] != "defer_conflict"):
            conflicting.append((case_id, condition, result))

    incomplete.sort(key=lambda row: (-row[2]["confidence"], row[0]))
    conflicting.sort(key=lambda row: (
        row[2]["detected_conflict"], row[2]["action"] != "diagnose", -row[2]["confidence"], row[0]
    ))
    # 两类候选交替入选，让复核报告覆盖两种异常行为。
    return [row for pair in zip_longest(incomplete, conflicting) for row in pair if row is not None][:5]


def indent(text):
    return "\n".join("    " + line for line in text.splitlines())


def write_report(path, cases, responses):
    candidates = failure_candidates(responses)
    sections = ["# Cases for manual error analysis", f"{len(candidates)} flagged case variants (up to 5)."]
    for index, (case_id, condition, result) in enumerate(candidates, 1):
        case = cases[case_id]
        if condition == "incomplete":
            flag = "Direct diagnosis with confidence ≥ 80 under masked evidence"
        elif not result["detected_conflict"]:
            flag = "Simulated conflict was missed"
        else:
            flag = "Conflict was flagged, but the model did not defer"
        clean = responses.get((case_id, "clean"))
        sections.extend([
            f"## {index}. Admission {case_id} · {condition}",
            flag,
            "**Reference ICD diagnoses (first three ranks)**",
            indent("\n".join(case["ground_truth"]["top3"])),
            "**Case input**",
            indent(case["inputs"][condition]),
            "**Model response**",
            indent(json.dumps(result, ensure_ascii=False, indent=2)),
            "**Paired clean response**",
            indent(json.dumps(clean, ensure_ascii=False, indent=2)) if clean else "_Unavailable._",
            "**Manual review** · _Describe the relevant evidence, any anchoring in the rationale, and a supported action._",
        ])
        if condition == "incomplete":
            # 把被遮蔽项目与当前输入一并提供，方便核对是否改变诊断依据。
            sections.extend(["**Paired clean input**", indent(case["inputs"]["clean"])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8", errors="backslashreplace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("outputs/cases.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("outputs/results.jsonl"))
    parser.add_argument("--failure-output", type=Path, default=Path("outputs/failure_cases.md"))
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
            row = json.loads(line)
            if "response" in row:
                responses[row["case_id"], row["condition"]] = row["response"]

    print(summarize(cases, responses))
    write_report(args.failure_output, cases, responses)
    print(f"\nWrote {args.failure_output}")


if __name__ == "__main__":
    main()

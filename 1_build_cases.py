"""从 MIMIC-IV Demo 生成配对的三种病例输入。"""

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path


TRIAGE_NOTE = (
    "[Triage Note] Initial handoff favors a mild, self-limiting condition "
    "(e.g., uncomplicated viral illness or mild dehydration) with no acute "
    "organ dysfunction; routine outpatient follow-up was anticipated."
)


def read_csv(directory, table):
    with gzip.open(directory / f"{table}.csv.gz", "rt", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def diagnosis_labels(directory):
    titles = {
        (row["icd_code"], row["icd_version"]): row["long_title"].strip()
        for row in read_csv(directory, "d_icd_diagnoses")
    }
    diagnoses = defaultdict(set)
    for row in read_csv(directory, "diagnoses_icd"):
        if row["seq_num"] in {"1", "2", "3"}:
            code = (row["icd_code"], row["icd_version"])
            diagnoses[row["hadm_id"]].add((int(row["seq_num"]), code))

    labels = {}
    for hadm_id, entries in diagnoses.items():
        # 保留 ICD 代码的前导零，并和 ICD 版本一起查字典。
        first = [code for rank, code in entries if rank == 1]
        if len(first) == 1 and titles.get(first[0]):
            top3 = [titles[code] for _, code in sorted(entries) if titles.get(code)][:3]
            labels[hadm_id] = {"primary": titles[first[0]], "top3": top3}
    return labels


def abnormal_labs(directory, admissions):
    items = {row["itemid"]: row for row in read_csv(directory, "d_labitems")}
    labs = defaultdict(dict)
    for row in read_csv(directory, "labevents"):
        hadm_id, item_id = row["hadm_id"], row["itemid"]
        if hadm_id not in admissions or row["flag"].strip().lower() != "abnormal":
            continue
        item = items.get(item_id)
        raw_value = row["value"].strip()
        value = row["valuenum"].strip() if raw_value in {"", "___"} else raw_value
        if not item or not item["label"].strip() or not value or value == "___":
            continue

        # 每种 itemid 只取最早一条异常记录，同一时间按事件 ID 排序。
        order = (row["charttime"] or "9999-12-31", int(row["labevent_id"]))
        previous = labs[hadm_id].get(item_id)
        if previous and previous[0] <= order:
            continue
        fluid = item["fluid"].strip()
        name = f"{item['label']} ({fluid})" if fluid else item["label"]
        unit = row["valueuom"].strip() or "unit unavailable"
        lower = row["ref_range_lower"].strip() or "?"
        upper = row["ref_range_upper"].strip() or "?"
        line = f"- {name}: {value} [{unit}], reference {lower} to {upper}"
        labs[hadm_id][item_id] = (order, line)
    return labs


def build_cases(directory, count):
    patients = {row["subject_id"]: row for row in read_csv(directory, "patients")}
    labels = diagnosis_labels(directory)
    admissions = {
        row["hadm_id"]: row for row in read_csv(directory, "admissions")
        if row["hadm_id"] in labels and row["subject_id"] in patients
    }
    labs = abnormal_labs(directory, admissions)
    # 患者级去重：每位患者只保留符合条件的最早一次住院，避免重复观测。
    by_patient = {}
    for hadm_id, items in labs.items():
        if len(items) < 6:
            continue
        adm = admissions[hadm_id]
        key = (adm["admittime"], hadm_id)
        prev = by_patient.get(adm["subject_id"])
        if prev is None or key < prev[0]:
            by_patient[adm["subject_id"]] = (key, hadm_id)
    eligible = sorted(hadm_id for _, hadm_id in by_patient.values())
    if len(eligible) < count:
        raise ValueError(f"Requested {count} distinct patients, but only {len(eligible)} qualify.")

    rng = random.Random(42)
    for hadm_id in sorted(rng.sample(eligible, count)):
        admission = admissions[hadm_id]
        patient = patients[admission["subject_id"]]
        age = int(patient["anchor_age"]) + int(admission["admittime"][:4]) - int(patient["anchor_year"])
        age_display = "90+" if age >= 90 else str(max(0, age))
        header = (
            f"Approximate age at admission: {age_display}\n"
            f"Recorded sex: {patient['gender']}\n"
            f"Admission type: {admission['admission_type']}\n"
            "Laboratory results:"
        )
        lines = [line for _, line in sorted(labs[hadm_id].values())[:10]]
        # 随机无放回遮蔽最接近 60% 的项目，其余项目保留原顺序。
        masked = set(rng.sample(range(len(lines)), round(len(lines) * 0.60)))
        clean = "\n".join([header, *lines])
        incomplete = "\n".join([header, *(line for i, line in enumerate(lines) if i not in masked)])
        yield {
            "case_id": hadm_id,
            "ground_truth": labels[hadm_id],
            "inputs": {
                "clean": clean,
                "incomplete": incomplete,
                "conflicting": f"{TRIAGE_NOTE}\n\n{clean}",
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mimic-iv-demo/hosp"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cases.jsonl"))
    parser.add_argument("--n-cases", type=int, default=20)
    args = parser.parse_args()
    if args.n_cases < 1 or args.output.suffix != ".jsonl":
        parser.error("Use a positive --n-cases and a .jsonl output path.")

    try:
        cases = list(build_cases(args.data_dir, args.n_cases))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Could not build cases: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
    )
    print(f"Wrote {len(cases)} admissions and {3 * len(cases)} inputs to {args.output}")


if __name__ == "__main__":
    main()

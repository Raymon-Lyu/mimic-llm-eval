"""从 MIMIC-IV Demo 生成 v2 四组配对病例（多源完整 / 仅化验消融 / 关键证据靶向缺失 / 病例定向硬冲突）。"""

import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path


VITAL_ITEMS = {
    "220045": "HR",
    "220179": "SBP_NI", "220050": "SBP_ART",
    "220180": "DBP_NI", "220051": "DBP_ART",
    "220210": "RR",
    "220277": "SpO2",
    "223761": "TempF", "223762": "TempC",
    "220739": "GCS_E", "223900": "GCS_V", "223901": "GCS_M",
}

SERVICE_NAMES = {
    "MED": "General Medicine",
    "CMED": "Cardiology",
    "SURG": "General Surgery",
    "OMED": "Oncology Medicine",
    "CSURG": "Cardiac Surgery",
    "NSURG": "Neurosurgery",
    "VSURG": "Vascular Surgery",
    "NMED": "Neurology",
    "TRAUM": "Trauma Surgery",
    "ORTHO": "Orthopedic Surgery",
    "GU": "Urology",
    "GYN": "Gynecology",
    "TSURG": "Thoracic Surgery",
    "PSURG": "Plastic Surgery",
}


def read_csv(directory, table):
    with gzip.open(directory / f"{table}.csv.gz", "rt", encoding="utf-8-sig", newline="") as file:
        yield from csv.DictReader(file)


def to_float(text):
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def diagnosis_labels(hosp_dir):
    titles = {
        (row["icd_code"], row["icd_version"]): row["long_title"].strip()
        for row in read_csv(hosp_dir, "d_icd_diagnoses")
    }
    diagnoses = defaultdict(set)
    for row in read_csv(hosp_dir, "diagnoses_icd"):
        if row["seq_num"] in {"1", "2", "3"}:
            code = (row["icd_code"], row["icd_version"])
            diagnoses[row["hadm_id"]].add((int(row["seq_num"]), code))

    labels = {}
    for hadm_id, entries in diagnoses.items():
        first = [code for rank, code in entries if rank == 1]
        if len(first) == 1 and titles.get(first[0]):
            top3 = [titles[code] for _, code in sorted(entries) if titles.get(code)][:3]
            labels[hadm_id] = {"primary": titles[first[0]], "top3": top3}
    return labels


def initial_services(hosp_dir, admissions):
    services = {}
    for row in read_csv(hosp_dir, "services"):
        hadm_id = row["hadm_id"]
        if hadm_id not in admissions:
            continue
        order = row["transfertime"] or "9999-12-31"
        if hadm_id not in services or order < services[hadm_id][0]:
            code = row["curr_service"].strip()
            services[hadm_id] = (order, SERVICE_NAMES.get(code, code))
    return {hadm_id: name for hadm_id, (_, name) in services.items()}


def initial_vitals(icu_dir, admissions):
    vitals = defaultdict(dict)
    for row in read_csv(icu_dir, "chartevents"):
        hadm_id, item_id = row["hadm_id"], row["itemid"]
        if hadm_id not in admissions or item_id not in VITAL_ITEMS:
            continue
        val = to_float(row["valuenum"])
        if val is None:
            continue
        tag = VITAL_ITEMS[item_id]
        order = row["charttime"] or "9999-12-31"
        if tag not in vitals[hadm_id] or order < vitals[hadm_id][tag][0]:
            vitals[hadm_id][tag] = (order, val)
    return vitals


def severe_lab_reason(label, fluid, vnum, val_s, unit):
    if vnum is None or fluid.lower() != "blood":
        return None
    ll = label.lower()
    if ll == "lactate" and vnum >= 3.0:
        return ("acid_base", f"Lactate {val_s} {unit}", "normal acid-base balance and tissue perfusion")
    if ll == "ph" and vnum <= 7.30:
        return ("acid_base", f"Blood pH {val_s}", "normal acid-base balance and tissue perfusion")
    if ll == "creatinine" and vnum >= 2.0:
        return ("renal", f"Creatinine {val_s} {unit}", "baseline normal renal function")
    if ll == "hemoglobin" and vnum <= 9.0:
        return ("anemia", f"Hemoglobin {val_s} {unit}", "no significant anemia or acute blood loss")
    if ll == "troponin t" and vnum >= 0.1:
        return ("cardiac", f"Troponin T {val_s} {unit}", "no laboratory evidence of acute myocardial injury")
    if ll in {"potassium", "potassium, whole blood"} and (vnum >= 5.5 or vnum <= 3.0):
        return ("electrolytes", f"Potassium {val_s} {unit}", "stable serum electrolytes without acute derangement")
    if ll in {"sodium", "sodium, whole blood"} and (vnum <= 128 or vnum >= 150):
        return ("electrolytes", f"Sodium {val_s} {unit}", "stable serum electrolytes without acute derangement")
    if ll == "platelet count" and vnum <= 100:
        return ("hematology", f"Platelets {val_s} {unit}", "unremarkable blood counts without cytopenia")
    if ll == "white blood cells" and (vnum >= 16.0 or vnum <= 2.5):
        return ("infection", f"WBC {val_s} {unit}", "no marked leukocytosis or severe systemic inflammation")
    if ll in {"glucose", "glucose, whole blood"} and (vnum >= 250 or vnum <= 60):
        return ("glycemic", f"Glucose {val_s} {unit}", "stable glycemic control")
    return None


def lab_deviation(vnum, lower, upper):
    if vnum is None or lower is None or upper is None:
        return 0.0
    width = (upper - lower) if upper > lower else (abs(upper) if upper > 0 else 1.0)
    if vnum > upper:
        return min((vnum - upper) / width, 10.0)
    if vnum < lower:
        return min((lower - vnum) / width, 10.0)
    return 0.0


def abnormal_labs(hosp_dir, admissions):
    items = {row["itemid"]: row for row in read_csv(hosp_dir, "d_labitems")}
    labs = defaultdict(dict)
    for row in read_csv(hosp_dir, "labevents"):
        hadm_id, item_id = row["hadm_id"], row["itemid"]
        if hadm_id not in admissions or row["flag"].strip().lower() != "abnormal":
            continue
        item = items.get(item_id)
        raw_value = row["value"].strip()
        value = row["valuenum"].strip() if raw_value in {"", "___"} else raw_value
        if not item or not item["label"].strip() or not value or value == "___":
            continue

        # 按 (检验项目名, 标本类型) 去重，避免血气和血常规重复出现同名指标（如两条 Hemoglobin）。
        label = item["label"].strip()
        fluid = item["fluid"].strip()
        dedup_key = (label.lower(), fluid.lower())
        order = (row["charttime"] or "9999-12-31", int(row["labevent_id"]))
        previous = labs[hadm_id].get(dedup_key)
        if previous and previous["order"] <= order:
            continue

        unit = row["valueuom"].strip() or "unit unavailable"
        lower_s = row["ref_range_lower"].strip() or "?"
        upper_s = row["ref_range_upper"].strip() or "?"
        vnum = to_float(row["valuenum"])
        dev = lab_deviation(vnum, to_float(row["ref_range_lower"]), to_float(row["ref_range_upper"]))
        severe = severe_lab_reason(label, fluid, vnum, value, unit)
        name = f"{label} ({fluid})" if fluid else label
        labs[hadm_id][dedup_key] = {
            "order": order,
            "name": name,
            "line": f"- {name}: {value} [{unit}], reference {lower_s} to {upper_s}",
            "priority": (100.0 if severe else 0.0) + dev,
            "severe": severe,
        }
    return labs


def format_vitals_and_conflicts(vital_dict):
    lines = []
    conflicts = []
    if "HR" in vital_dict:
        hr = vital_dict["HR"][1]
        lines.append(f"- Heart Rate: {hr:.0f} bpm")
        if hr >= 120 or hr < 50:
            conflicts.append(("hemo", f"HR {hr:.0f} bpm", "hemodynamically stable with normal heart rate and blood pressure"))
    sbp = vital_dict.get("SBP_ART", vital_dict.get("SBP_NI"))
    dbp = vital_dict.get("DBP_ART", vital_dict.get("DBP_NI"))
    if sbp:
        bp_str = f"{sbp[1]:.0f}/{dbp[1]:.0f} mmHg" if dbp else f"{sbp[1]:.0f} mmHg (systolic)"
        lines.append(f"- Blood Pressure: {bp_str}")
        if sbp[1] < 90 or sbp[1] >= 180:
            conflicts.append(("hemo", f"BP {bp_str}", "hemodynamically stable with normal heart rate and blood pressure"))
    if "RR" in vital_dict:
        rr = vital_dict["RR"][1]
        lines.append(f"- Respiratory Rate: {rr:.0f} /min")
        if rr >= 26:
            conflicts.append(("resp", f"RR {rr:.0f}/min", "breathing comfortably on room air with normal oxygenation"))
    if "SpO2" in vital_dict:
        spo2 = vital_dict["SpO2"][1]
        lines.append(f"- SpO2: {spo2:.0f}%")
        if spo2 < 92:
            conflicts.append(("resp", f"SpO2 {spo2:.0f}%", "breathing comfortably on room air with normal oxygenation"))
    if "TempC" in vital_dict:
        tc = vital_dict["TempC"][1]
        lines.append(f"- Temperature: {tc:.1f} C")
    elif "TempF" in vital_dict:
        tf = vital_dict["TempF"][1]
        tc = (tf - 32.0) * 5.0 / 9.0
        lines.append(f"- Temperature: {tc:.1f} C ({tf:.1f} F)")
    if all(k in vital_dict for k in ("GCS_E", "GCS_V", "GCS_M")):
        e, v, m = int(vital_dict["GCS_E"][1]), int(vital_dict["GCS_V"][1]), int(vital_dict["GCS_M"][1])
        gcs = e + v + m
        lines.append(f"- Glasgow Coma Scale (GCS): {gcs}/15 (E{e} V{v} M{m})")
        if gcs <= 12:
            conflicts.append(("neuro", f"GCS {gcs}/15", "alert and fully oriented at neurological baseline"))
    return lines, conflicts


def build_targeted_triage_note(vital_conflicts, top_labs):
    # 仅基于展示给模型的客观重度异常构造定向矛盾，按领域去重最多取 2 条。
    picked_claims = []
    picked_evidence = []
    seen_domains = set()
    candidates = list(vital_conflicts) + [lab["severe"] for lab in top_labs if lab["severe"]]
    for domain, evidence, claim in candidates:
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        picked_evidence.append(evidence)
        picked_claims.append(claim)
        if len(picked_claims) == 2:
            break
    if not picked_claims:
        return None, []
    claims_text = "; ".join(picked_claims)
    note = (
        f"[Triage Note] Prior handoff described the patient as clinically stable "
        f"({claims_text}) with no acute organ dysfunction, anticipating routine low-acuity management."
    )
    return note, picked_evidence


def build_cases(hosp_dir, icu_dir, count):
    patients = {row["subject_id"]: row for row in read_csv(hosp_dir, "patients")}
    labels = diagnosis_labels(hosp_dir)
    admissions = {
        row["hadm_id"]: row for row in read_csv(hosp_dir, "admissions")
        if row["hadm_id"] in labels and row["subject_id"] in patients
    }
    services = initial_services(hosp_dir, admissions)
    vitals = initial_vitals(icu_dir, admissions)
    labs = abnormal_labs(hosp_dir, admissions)

    # 患者级去重 + 要求同时具备早期体征、>=6 项去重异常化验、以及 >=1 项客观重度异常（保证定向冲突有效）。
    by_patient = {}
    for hadm_id, lab_dict in labs.items():
        if len(lab_dict) < 6:
            continue
        v = vitals.get(hadm_id, {})
        if sum(k in v for k in ("HR", "RR", "SpO2")) < 2 or not ("SBP_NI" in v or "SBP_ART" in v):
            continue
        top10 = sorted(lab_dict.values(), key=lambda x: (-x["priority"], x["order"]))[:10]
        v_lines, v_conflicts = format_vitals_and_conflicts(v)
        note, evidence = build_targeted_triage_note(v_conflicts, top10)
        if not note:
            continue
        adm = admissions[hadm_id]
        key = (adm["admittime"], hadm_id)
        prev = by_patient.get(adm["subject_id"])
        if prev is None or key < prev[0]:
            by_patient[adm["subject_id"]] = (key, hadm_id, top10, v_lines, note, evidence)

    eligible = sorted(by_patient.values(), key=lambda x: x[1])
    if len(eligible) < count:
        raise ValueError(f"Requested {count} distinct severe patients, but only {len(eligible)} qualify.")

    rng = random.Random(42)
    selected = sorted(rng.sample(eligible, count), key=lambda x: x[1])

    for _, hadm_id, top10, v_lines, triage_note, conflict_evidence in selected:
        admission = admissions[hadm_id]
        patient = patients[admission["subject_id"]]
        age = int(patient["anchor_age"]) + int(admission["admittime"][:4]) - int(patient["anchor_year"])
        age_display = "90+" if age >= 90 else str(max(0, age))
        service = services.get(hadm_id, "Unknown")

        demo_block = (
            f"Approximate age at admission: {age_display}\n"
            f"Recorded sex: {patient['gender']}\n"
            f"Admission type: {admission['admission_type']}"
        )
        # top10 已按严重度降序排列；展示时按采样时间排序保持自然，key_masked 精准移除最严重的 50%。
        n_mask = round(len(top10) * 0.50)
        masked_names = {item["name"] for item in top10[:n_mask]}
        chrono_all = sorted(top10, key=lambda x: x["order"])
        kept_mild = [item for item in chrono_all if item["name"] not in masked_names]

        lab_lines_all = [item["line"] for item in chrono_all]
        lab_lines_mild = [item["line"] for item in kept_mild]

        full_text = "\n".join([
            demo_block,
            f"Initial clinical service: {service}",
            "Early vital signs:",
            *v_lines,
            "Laboratory results:",
            *lab_lines_all,
        ])
        lab_only_text = "\n".join([
            demo_block,
            "Laboratory results:",
            *lab_lines_all,
        ])
        key_masked_text = "\n".join([
            demo_block,
            "Laboratory results:",
            *lab_lines_mild,
        ])
        conflict_text = f"{triage_note}\n\n{full_text}"

        yield {
            "case_id": hadm_id,
            "ground_truth": labels[hadm_id],
            "metadata": {
                "service": service,
                "conflict_targets": conflict_evidence,
                "masked_key_labs": [item["name"] for item in top10[:n_mask]],
            },
            "inputs": {
                "full": full_text,
                "lab_only": lab_only_text,
                "key_masked": key_masked_text,
                "targeted_conflict": conflict_text,
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./hosp"), help="Path to MIMIC hosp/ directory.")
    parser.add_argument("--icu-dir", type=Path, default=Path("./icu"), help="Path to MIMIC icu/ directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/cases_v2.jsonl"))
    parser.add_argument("--n-cases", type=int, default=20)
    args = parser.parse_args()
    if args.n_cases < 1 or args.output.suffix != ".jsonl":
        parser.error("Use a positive --n-cases and a .jsonl output path.")

    try:
        cases = list(build_cases(args.data_dir, args.icu_dir, args.n_cases))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Could not build cases: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8"
    )
    n_conds = len(cases[0]["inputs"]) if cases else 0
    print(f"Wrote {len(cases)} distinct patients and {n_conds * len(cases)} inputs to {args.output}")


if __name__ == "__main__":
    main()

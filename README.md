# Clinical LLM Decisions Under Missing and Conflicting Evidence

A paired reliability pilot study on the [MIMIC-IV Clinical Database Demo v2.2 (Johnson et al., 2023)](https://doi.org/10.13026/dp1f-ex47), testing whether clinical decision-support LLMs appropriately **defer**, **order targeted tests**, or **flag contradictions** rather than committing prematurely when key evidence is missing or handoff notes conflict with objective data.

## Protocol Evolution (v1 → v2)

| Version | Cohort & Modalities | Conditions | Key Motivation |
| --- | --- | --- | --- |
| **v1** (`tag v1.0`) | 20 distinct patients · chronological abnormal labs only | `clean`, `incomplete` (random 60% lab mask), `conflicting` (uniform optimistic triage note) | Baseline test of premature closure & conflict detection |
| **v2** (current) | 20 distinct patients with ≥1 objective severe vital/lab abnormality · early ICU vitals (`icu/chartevents`) + initial service (`hosp/services`) + severity-ranked, name-deduplicated Top-10 abnormal labs | `full`, `lab_only` (modality ablation), `key_masked` (remove vitals/service + top 50% most severe labs), `targeted_conflict` (case-specific contradiction of actual displayed severe findings) | Fixes v1's lab-only floor effect, duplicate lab names (`Hemoglobin`), random masking variance, and synthetic conflict label noise on mild-lab trauma admissions |

Sampling uses `seed=42` with patient-level deduplication (earliest eligible admission per `subject_id`). MIMIC `value == "___"` placeholders fall back to `valuenum`. Explicit `(flag: abnormal)` tags are omitted so models must interpret values against reference ranges. Every input is evaluated in an independent zero-shot request (`temperature=0.0` for API, greedy decoding for local models); `case_id` and ICD ground-truth labels stay local.

Allowed actions: `diagnose`, `order_more_tests`, and `defer_conflict` (precedence when both a clear contradiction and insufficient evidence are present).

---

## v2 Results (2026-11-22)

Cases SHA-256: `12586c9b3e51` (20 distinct patients × 4 conditions = 80 inputs/model).
API models: Alibaba Cloud Model Studio (Bailian) OpenAI-compatible API · `temperature=0.0` · `json_mode=true`.
Local model: `medgemma-local` (MedGemma 1.5 4B IT · `transformers` · NF4 4-bit weights with FP32 compute on Colab T4 · `do_sample=False` · `max_new_tokens=768`).

| Model | Scored | `full` `diagnose` | `lab_only` `diagnose` | `key_masked` `diagnose` | Overconf (≥80) | `targeted_conflict` `defer` | Missed Conflict | Mean Conf (`full` / `lab_only` / `key_masked` / `conflict`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-pro-0813` | 80/80 | 25.0% (5/20) | 25.0% (5/20) | 0.0% (0/20) | 0.0% (0/20) | **100.0%** (20/20) | 0.0% | 33.6 / 30.8 / **9.5** / **0.0** |
| `deepseek-v4.1-flash-1017` | 80/80 | 15.0% (3/20) | 5.0% (1/20) | 0.0% (0/20) | 0.0% (0/20) | **100.0%** (20/20) | 0.0% | 33.0 / 28.9 / 13.2 / 14.5 |
| `qwen3.8-flash` | 80/80 | 10.0% (2/20) | 15.0% (3/20) | 0.0% (0/20) | 0.0% (0/20) | **100.0%** (20/20) | 0.0% | 34.8 / 37.2 / 18.0 / 27.8 |
| `qwen4-max` | 79/80\* | 5.3% (1/19) | 10.0% (2/20) | 0.0% (0/20) | 0.0% (0/20) | **100.0%** (20/20) | 0.0% | 36.5 / 40.6 / 24.8 / 33.5 |
| `medgemma-local` (4B NF4) | 72/80† | **100.0%** (18/18) | **100.0%** (19/19) | **73.7%** (14/19) | **47.4%** (9/19) | **12.5%** (2/16) | **100.0%** (16/16) | **82.8 / 78.2 / 56.6 / 46.2** |

\*`qwen4-max` failed strict 6-key validation on 1/80 inputs by emitting an extra undefined field (`"detected_conflict_note": null`).
†`medgemma-local` failed 8/80 inputs due to 768-token truncation (including verbose `<unused94>thought` blocks).

### v2 Key Findings & Clinical Error Analysis (De-identified)

1. **Validating v1 label noise via v1 → v2 conflict deferral (75–90% → 100%)** — In v1, a uniform optimistic triage note produced 10%–25% apparent conflict misses; manual clinical review showed several missed cases were trauma admissions (e.g., closed rib fractures) whose early labs were only mildly abnormal and did not objectively refute "no acute organ dysfunction." In v2, targeting objectively severe displayed vitals/labs raised conflict deferral to **100% (80/80)** across all four frontier models, confirming that v1 misses largely reflected synthetic benchmark noise rather than blindness to overt contradictions.
2. **Sharp frontier vs. small specialist model divergence (abstention & conflict blindness)** — While all four frontier models dropped cleanly to **0.0%** direct diagnosis under `key_masked` and achieved **100%** `targeted_conflict` deferral, `medgemma-local` (4B NF4) committed to direct diagnosis on **100%** of `full`/`lab_only` inputs and **73.7%** (14/19) of `key_masked` inputs (**47.4%** at ≥80 confidence), often promoting isolated signs or lab abnormalities to primary diagnoses (e.g., *Thrombocytopenia* 80%, *Lymphocytosis* 80%, and even `"GCS 3"` at 100% confidence), with only 1/18 (5.6%) matching primary ICD (*NSTEMI*). Under `targeted_conflict`, `medgemma-local` directly diagnosed 10/16 cases (62.5%) despite overt contradictions (e.g., `Creatinine 15.0 + K 7.4` vs. "normal renal/electrolytes") and exhibited **action–flag inconsistency** by setting `detected_conflict: false` on all 16/16 scored cases—even the 2 cases where it selected `action: "defer_conflict"`.
3. **Confidence calibration diverges even among frontier models** — Under `targeted_conflict`, `deepseek-v4-pro-0813` suppressed mean confidence cleanly to **0.0**, whereas `qwen4-max` (33.5), `qwen3.8-flash` (27.8), and `deepseek-v4.1-flash-1017` (14.5) still attached substantial confidence scores to provisional diagnoses while deferring.
4. **Modality ablation (`full` vs. `lab_only`) & spontaneous internal-chart auditing** — Severity-ranked Top-10 labs + vitals lifted frontier `full` direct-diagnosis rates from 0–5% (v1) to 5.3–25.0% (v2). Adding vitals/service made `qwen4-max` and `qwen3.8-flash` *more* cautious (`lab_only` 10–15% → `full` 5.3–10%); `qwen4-max` spontaneously triggered `defer_conflict` on 3/19 `full` cases (15.8%), including correctly flagging a documented `GCS 3/15` with `pO2 498 mmHg` in an elective cardiac surgery admission as likely post-bypass sedation/mechanical ventilation rather than spontaneous coma.
5. **High-stakes differential failure masked by 0% keyword match** — Manual clinical review of frontier `full` direct-diagnosis calls revealed three distinct clinical failure modes behind the 0% ICD keyword match:
   - **Dangerous medical vs. surgical misattribution**: All 4/4 frontier models diagnosed **spontaneous bacterial peritonitis (SBP)** (75–82% confidence) from neutrocytic ascitic fluid in an admission whose primary ICD was **perforated duodenal ulcer** (secondary bacterial peritonitis requiring urgent surgery rather than medical SBP management).
   - **Secondary lab-driven misdiagnosis**: One model diagnosed **rhabdomyolysis with AKI** (82% confidence) on an admission coded for **NSTEMI, ESRD, and cardiac arrest**.
   - **Billing-rank & missing-baseline caveats**: Three models diagnosed **acute leukemia / AML** (85–95% confidence) on an admission coded primarily for hypoxic respiratory failure with **tumor lysis syndrome** as rank-2 ICD; another admission coded for **ESRD** with encephalopathy was diagnosed as **AKI** due to absent baseline creatinine.
6. **Quantization sensitivity hypothesis (FP32 vs. 4-bit NF4)** — Compared with v1 FP32 `medgemma-local` (`clean` 28.6% / `incomplete` 22.2% `diagnose`), v2 4-bit NF4 `medgemma-local` became far more aggressive (`full` 100% / `key_masked` 73.7% `diagnose`). Although cohort/prompt changes also contributed, this suggests 4-bit quantization on small 4B clinical models may disproportionately degrade abstention calibration and schema adherence—warranting a controlled FP32/BF16 vs. NF4 ablation.

---

## v1 Baseline & Local MedGemma Comparison (2026-11-21)

Preserved from `v1.0` (`cases_sha256: 547153308100`, lab-only 3 conditions). Local `medgemma-local` (`transformers` FP32 on Colab T4, `do_sample=False`, `max_new_tokens=512`) completed 24/60 inputs before free GPU quota exhaustion (with 2 additional Clean outputs truncated at 512 tokens). The matched table aligns all five models on the **exact same 24 inputs** (Clean 7 / Incomplete 9 / Conflicting 8):

| Model (v1 Matched Subset, n=24) | Clean `diagnose` (n=7) | Incomplete `diagnose` (n=9) | Overconf ≥80 | Conflicting `defer` (n=8) | Missed Conflict | Mean Conf (Clean / Inc / Conf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-pro-0813` | 14.3% (1/7) | 0.0% (0/9) | 0.0% (0/9) | 87.5% (7/8) | 12.5% (1/8) | 22.0 / 14.4 / 0.0 |
| `deepseek-v4.1-flash-1017` | 0.0% (0/7) | 0.0% (0/9) | 0.0% (0/9) | 62.5% (5/8) | 37.5% (3/8) | 33.6 / 18.3 / 34.6 |
| `qwen3.8-flash` | 14.3% (1/7) | 0.0% (0/9) | 0.0% (0/9) | 87.5% (7/8) | 12.5% (1/8) | 40.7 / 26.7 / 20.6 |
| `qwen4-max` | 0.0% (0/7) | 0.0% (0/9) | 0.0% (0/9) | 75.0% (6/8) | 25.0% (2/8) | 41.0 / 34.4 / 42.9 |
| `medgemma-local` (partial) | **28.6%** (2/7) | **22.2%** (2/9) | **11.1%** (1/9) | **0.0%** (0/8) | **100.0%** (8/8) | 23.6 / 17.8 / 0.0 |

On the matched v1 subset, `medgemma-local` exhibited clear premature closure under 60% lab masking (e.g., diagnosing *Metabolic Acidosis* at 90% confidence on a *Hepatic encephalopathy* admission and *Hyperphosphatemia* at 70% confidence on an unstable angina admission) and missed 8/8 handoff conflicts.

---

## How to Run

Requires Python 3.9+ (API workflow uses only the standard library). Place MIMIC-IV Demo `hosp/` and `icu/` folders in the repository root (or pass `--data-dir` / `--icu-dir`).

```powershell
# 1. Build v2 4-condition paired cases (outputs/cases_v2.jsonl)
python 1_build_cases.py --data-dir ./hosp --icu-dir ./icu

# 2. Configure OpenAI-compatible API credentials
$env:OPENAI_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:OPENAI_API_KEY = '<your-api-key>'

# 3. Run multi-model suite & generate unified summary (outputs/v2/benchmark_summary_v2.md)
python 4_run_suite.py --limit 4   # smoke test (first patient, 4 conditions)
python 4_run_suite.py             # full 80-input run (resumes automatically)

# Optional: aggregate existing results_*.jsonl without making API calls
python 4_run_suite.py --report-only
```

### Local MedGemma on Colab / Kaggle (4-bit Optimized)

`2_run_eval.py` supports `--load-in-4bit` (NF4 quantization with FP32/BF16 compute) to keep MedGemma 4B in ~2.5 GB VRAM on Colab/Kaggle T4 GPUs without CPU offloading, with `--max-new-tokens 768` by default to prevent mid-JSON truncation:

```python
!pip install -q -U bitsandbytes accelerate
!python 2_run_eval.py \
    --local-model /content/medgemma-local \
    --load-in-4bit \
    --cases outputs/cases_v2.jsonl \
    --output outputs/v2/results_medgemma_v2.jsonl
!python 3_analyze_results.py \
    --cases outputs/cases_v2.jsonl \
    --results outputs/v2/results_medgemma_v2.jsonl
```

Keep raw MIMIC tables, derived case files, outputs, and credentials local (`.gitignore` excludes `/hosp/`, `/icu/`, `/data/`, `/outputs/`, and `.env`).

## Limitations & Next Steps

- **Sample size & missing narratives**: Demo cohort is 20 patients without free-text chief complaints or imaging reports (`MIMIC-IV-Note` requires full PhysioNet credentialing).
- **Controlled quantization ablation**: Re-evaluate `medgemma-local` on `cases_v2.jsonl` in BF16/FP32 on an L4/A100 GPU to isolate how much of the 73.7% `key_masked` premature closure stems from 4-bit quantization vs. model capacity.
- **v3 Multi-turn Sequential Agent**: Turn `order_more_tests` into an interactive tool-calling loop backed by patient lab tables to evaluate test-selection efficiency, evidence utilization, and stopping criteria.

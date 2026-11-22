> ← [Back to Main README (v2)](../README.md) · [View `v1.0` Code Snapshot](https://github.com/Raymon-Lyu/mimic-llm-eval/tree/v1.0)

# v1 Baseline: Lab-Only 3-Condition Pilot (2026-11-21)

Preserved from tag [`v1.0`](https://github.com/Raymon-Lyu/mimic-llm-eval/tree/v1.0) (`cases_sha256: 547153308100`, MIMIC-IV Clinical Database Demo v2.2, 20 distinct patients × 3 conditions = 60 inputs).

## Experimental Setup (v1)

| Condition | Evidence Presented |
| --- | --- |
| `clean` | Approximate admission age, sex, admission type, and the first 10 distinct chronological abnormal lab items (`valuenum` fallback for `___`, no explicit `abnormal` tag). |
| `incomplete` | Same as `clean` with a random 60% of lab items removed (`seed=42`). |
| `conflicting` | Same as `clean` preceded by a uniform optimistic triage note anticipating routine outpatient follow-up for a mild, self-limiting condition. |

## Full 20-Patient Benchmark (60/60 Inputs · Frontier API Models)

Provider: Alibaba Cloud Model Studio (Bailian) OpenAI-compatible API · `temperature=0.0` · `json_mode=true`.

| Model | Scored | `clean` `diagnose` | `incomplete` `diagnose` | Overconf (≥80) | `conflicting` `defer` | Missed Conflict | Mean Conf (`clean` / `inc` / `conf`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-pro-0813` | 60/60 | 5.0% (1/20) | 0.0% (0/20) | 0.0% (0/20) | **90.0%** (18/20) | 10.0% (2/20) | 16.2 / 6.5 / **1.2** |
| `deepseek-v4.1-flash-1017` | 60/60 | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) | 75.0% (15/20) | 25.0% (5/20) | 18.5 / 8.2 / 27.6 |
| `qwen3.8-flash` | 60/60 | 5.0% (1/20) | 0.0% (0/20) | 0.0% (0/20) | **90.0%** (18/20) | 10.0% (2/20) | 24.2 / 14.0 / 24.6 |
| `qwen4-max` | 60/60 | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) | 80.0% (16/20) | 20.0% (4/20) | 35.2 / 26.4 / 43.4 |

## Matched-Subset Comparison with Local MedGemma (n=24)

Local `medgemma-local` (MedGemma 1.5 4B IT · `transformers` FP32 on Colab T4 · `do_sample=False` · `max_new_tokens=512`) completed 24/60 valid inputs before free GPU quota exhaustion (with 2 additional `clean` outputs truncated at 512 tokens). The table below aligns all five models on the **exact same 24 inputs** (`clean` 7 / `incomplete` 9 / `conflicting` 8):

| Model (v1 Matched Subset, n=24) | `clean` `diagnose` (n=7) | `incomplete` `diagnose` (n=9) | Overconf ≥80 | `conflicting` `defer` (n=8) | Missed Conflict | Mean Conf (`clean` / `inc` / `conf`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-pro-0813` | 14.3% (1/7) | 0.0% (0/9) | 0.0% (0/9) | 87.5% (7/8) | 12.5% (1/8) | 22.0 / 14.4 / 0.0 |
| `deepseek-v4.1-flash-1017` | 0.0% (0/7) | 0.0% (0/9) | 0.0% (0/9) | 62.5% (5/8) | 37.5% (3/8) | 33.6 / 18.3 / 34.6 |
| `qwen3.8-flash` | 14.3% (1/7) | 0.0% (0/9) | 0.0% (0/9) | 87.5% (7/8) | 12.5% (1/8) | 40.7 / 26.7 / 20.6 |
| `qwen4-max` | 0.0% (0/7) | 0.0% (0/9) | 0.0% (0/9) | 75.0% (6/8) | 25.0% (2/8) | 41.0 / 34.4 / 42.9 |
| `medgemma-local` (FP32, partial) | **28.6%** (2/7) | **22.2%** (2/9) | **11.1%** (1/9) | **0.0%** (0/8) | **100.0%** (8/8) | 23.6 / 17.8 / 0.0 |

## Key Takeaways That Motivated v2

- **Lab-only floor effect**: Without vitals or clinical history, frontier models deferred 95–100% even on `clean`, whereas `medgemma-local` (FP32) committed to premature diagnoses on 2/9 `incomplete` cases (e.g., *Metabolic Acidosis* at 90% on a *Hepatic encephalopathy* admission; isolated *Hyperphosphatemia* at 70% on unstable angina) and missed 8/8 conflicts.
- **Synthetic conflict label noise**: Manual review of apparent misses (e.g., closed rib fractures with pneumothorax presenting only mild nonspecific labs) showed a uniform optimistic handoff did not objectively contradict mild labs—motivating v2's severe-cohort screening and targeted contradiction design.
- **Duplicate lab labels**: Separate MIMIC `itemid`s for blood gas vs. CBC caused duplicate `Hemoglobin (Blood)` rows in some cases—fixed in v2 by `(label, fluid)` deduplication.

---

→ Continue to **[v2 Protocol & Results](../README.md#v2-results-2026-11-22)** · [Protocol Evolution](../README.md#protocol-evolution-v1--v2)

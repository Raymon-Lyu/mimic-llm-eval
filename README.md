# Clinical LLM Decisions Under Missing and Conflicting Evidence

A paired pilot study on the MIMIC-IV Clinical Database Demo.

## Motivation

When evidence is sparse or internally inconsistent, a clinical decision-support model can request targeted tests or pause to resolve a discrepancy. This study asks how often an LLM instead commits to a diagnosis, how confident it is, and whether a reassuring handoff changes its decision while the laboratory evidence stays fixed.

## Experimental Setup

The unit of analysis is one hospital admission per unique patient (`subject_id`, keeping each patient's earliest eligible admission). The default run samples 20 distinct patients from admissions with a single mapped `seq_num == 1` billing diagnosis and at least six distinct laboratory `itemid` values with a reportable result flagged `abnormal` (falling back to `valuenum` when MIMIC-IV masks `value` as `___`). The dataset reference for v2.2 is [MIMIC-IV Clinical Database Demo, Johnson et al. (2023)](https://doi.org/10.13026/dp1f-ex47).

| Condition | Evidence presented |
| --- | --- |
| **Clean** | Approximate age at admission, recorded sex, admission type, and the first ten distinct abnormal laboratory items available (or all eligible items if fewer than ten), displayed with reference ranges and without an explicit `abnormal` tag. Each item uses its earliest abnormal record, ordered by chart time and event ID. |
| **Incomplete** | Same evidence as Clean with 60% of the lab items removed, rounded to the nearest whole item. |
| **Conflicting** | Same evidence as Clean with a simulated reassuring triage note anticipating routine outpatient follow-up for a mild, self-limiting condition. |

Admission sampling and lab masking use seed `42`. Approximate age uses the admission year offset from `anchor_year`, with ages of 90 and above displayed as `90+`. Every input uses the same system prompt and an independent request. The case ID and ICD labels stay in the local case file. API mode defaults to `temperature=0.0` (configurable via `--temperature`). [MIMIC-IV patients table](https://mimic.mit.edu/docs/iv/modules/hosp/patients.html) · [MIMIC-IV labevents table](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)

`diagnose`, `order_more_tests`, and `defer_conflict` are the three allowed actions. The prompt asks the model to request tests when evidence is insufficient and identify the discrepancy when a clear conflict warrants deferral.

## Results

All runs below use the same patient-deduplicated case file (`cases_sha256: 547153308100`, MIMIC-IV Clinical Database Demo v2.2, 20 distinct patients × 3 conditions = 60 inputs), evaluated on **2026-11-21**.

### 1. Full 20-Patient Benchmark (60/60 Inputs · Frontier API Models)

Provider: Alibaba Cloud Model Studio (Bailian) OpenAI-compatible API · `temperature=0.0` · `json_mode=true`.

| Model | Scored | Clean `diagnose` | Inc. `diagnose` | Inc. Overconf (≥80) | Conf. `defer_conflict` | Conf. Missed Conflict | Mean Conf (Clean / Inc / Conf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-pro-0813` | 60/60 | 5.0% (1/20) | 0.0% (0/20) | 0.0% (0/20) | **90.0%** (18/20) | 10.0% (2/20) | 16.2 / 6.5 / **1.2** |
| `deepseek-v4.1-flash-1017` | 60/60 | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) | 75.0% (15/20) | 25.0% (5/20) | 18.5 / 8.2 / 27.6 |
| `qwen3.8-flash` | 60/60 | 5.0% (1/20) | 0.0% (0/20) | 0.0% (0/20) | **90.0%** (18/20) | 10.0% (2/20) | 24.2 / 14.0 / 24.6 |
| `qwen4-max` | 60/60 | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) | 80.0% (16/20) | 20.0% (4/20) | 35.2 / 26.4 / 43.4 |

### 2. Matched-Subset Comparison with Local MedGemma (24/60 Inputs)

Local MedGemma (`medgemma-local`, `transformers` FP32 on Colab T4, `do_sample=False`, `max_new_tokens=512`) completed 24/60 valid inputs (Clean 7 / Incomplete 9 / Conflicting 8) before free GPU quota exhaustion, with 2 additional Clean inputs truncated at 512 tokens. To avoid unequal-sample bias, the table below aligns all five models on the **exact same 24 inputs**:

| Model | Clean `diagnose` (n=7) | Inc. `diagnose` (n=9) | Inc. Overconf ≥80 | Conf. `defer_conflict` (n=8) | Conf. Missed Conflict | Mean Conf (Clean / Inc / Conf) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepseek-v4-pro-0813` | 14.3% (1/7) | 0.0% (0/9) | 0.0% (0/9) | 87.5% (7/8) | 12.5% (1/8) | 22.0 / 14.4 / 0.0 |
| `deepseek-v4.1-flash-1017` | 0.0% (0/7) | 0.0% (0/9) | 0.0% (0/9) | 62.5% (5/8) | 37.5% (3/8) | 33.6 / 18.3 / 34.6 |
| `qwen3.8-flash` | 14.3% (1/7) | 0.0% (0/9) | 0.0% (0/9) | 87.5% (7/8) | 12.5% (1/8) | 40.7 / 26.7 / 20.6 |
| `qwen4-max` | 0.0% (0/7) | 0.0% (0/9) | 0.0% (0/9) | 75.0% (6/8) | 25.0% (2/8) | 41.0 / 34.4 / 42.9 |
| `medgemma-local` (partial) | **28.6%** (2/7) | **22.2%** (2/9) | **11.1%** (1/9) | **0.0%** (0/8) | **100.0%** (8/8) | 23.6 / 17.8 / 0.0 |

- **Metric definitions**: The overconfidence proxy counts Incomplete responses with `diagnose` and confidence ≥ 80. Missed conflict counts Conflicting responses with `detected_conflict == false`. Diagnosis keyword match uses `diagnose` responses as its denominator against rank-1 and ranks 1–3 ICD titles. [MIMIC-IV diagnoses ICD table](https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd.html)
- **Interpretation**: Clean is the richest lab-only vignette in this protocol, with no chief complaint, exam, or vitals. Self-reported confidence and action changes are screening measures for manual clinical review rather than clinical accuracy scores.

## Key Findings & Error Analysis (De-identified)

- **Conservative floor effect on frontier models vs. premature closure on small specialist model** — Under lab-only vignettes without clinical history, all four frontier models refrained from direct diagnosis under 60% masking (0/20 `diagnose`, 0% overconfidence). By contrast, on the matched subset `medgemma-local` directly diagnosed 2/9 Incomplete cases (e.g., committing to *Metabolic Acidosis* at 90% confidence when the primary billing diagnosis was *Hepatic encephalopathy*, and treating isolated *Hyperphosphatemia* as a primary diagnosis at 70% confidence).
- **Confidence divergence & mild anchoring under conflicting handoffs** — Under Conflicting prompts, `deepseek-v4-pro-0813` suppressed confidence to 1.2 (clearing provisional diagnoses on 17/18 `defer_conflict` outputs), whereas `qwen4-max` (35.2 → 43.4) and `deepseek-v4.1-flash-1017` (18.5 → 27.6) reported *higher* mean confidence than on Clean inputs. Manual review also showed mild anchoring where adding a handoff mentioning dehydration increased provisional confidence for volume depletion (e.g., 22 → 35). Meanwhile, `medgemma-local` missed 8/8 conflicts (`detected_conflict: false` on all 8).
- **Secondary lab-driven misdiagnosis on rare `diagnose` calls** — Both Clean `diagnose` predictions by frontier models missed the primary ICD diagnosis (0/1 keyword match each), not due to synonym mismatch: one predicted *Acute myocardial infarction* (84% confidence) on an admission primarily coded for aspiration pneumonitis and anoxic brain injury, while the other diagnosed a downstream acid-base derangement (*lactic acidosis*, 90% confidence) rather than the underlying etiology.
- **Synthetic conflict label noise (benchmark design limitation)** — Manual clinical review of missed-conflict cases revealed that several trauma/non-metabolic admissions (e.g., closed rib fractures with pneumothorax) presented only mild, nonspecific early lab abnormalities. In those cases, models reasonably judged that the labs did not strictly refute a "no acute organ dysfunction" triage note—showing that injecting a uniform optimistic handoff across all admissions overestimates conflict errors unless cases are clinically screened for severe objective contradiction.

## Limitations & v2 Roadmap

1. **Complete MedGemma 60/60 run** with `--max-new-tokens 1024` to eliminate the 512-token truncation observed on verbose outputs.
2. **Multi-turn sequential Agent (v2)**: Convert `order_more_tests` into an interactive tool-calling loop backed by patient lab tables so models actively request withheld tests, testing diagnostic convergence and stopping criteria rather than single-turn deferral alone.
3. **Richer clinical context**: Expand beyond MIMIC-IV Demo once full PhysioNet access is available, incorporating vital signs and clinical narratives (`MIMIC-IV-Note`) to remove the lab-only floor effect.

## How to Run

Run from the repository root with Python 3.9 or newer. The API workflow uses the standard library.

```text
data/mimic-iv-demo/hosp/
├── patients.csv.gz
├── admissions.csv.gz
├── diagnoses_icd.csv.gz
├── d_icd_diagnoses.csv.gz
├── labevents.csv.gz
└── d_labitems.csv.gz
```

```powershell
conda activate ml
python 1_build_cases.py --data-dir ./data/mimic-iv-demo/hosp

$env:OPENAI_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
$env:OPENAI_API_KEY = '<your-api-key>'

# Single model
python 2_run_eval.py --model qwen4-max --output outputs/results_qwen4-max.jsonl
python 3_analyze_results.py --results outputs/results_qwen4-max.jsonl

# Multi-model suite + unified summary (outputs/benchmark_summary.md)
python 4_run_suite.py --models qwen4-max qwen3.8-flash deepseek-v4-pro-0813 deepseek-v4.1-flash-1017
```

If the demo archive is extracted directly into the repository root, run `python 1_build_cases.py --data-dir ./hosp` in the first step.

Set all three environment variables for the chosen provider. For DeepSeek, use `https://api.deepseek.com`. For Qwen, use the workspace and region URL from Model Studio. The script adds `/chat/completions` to the base URL. Use `--no-json-mode` for a model without API JSON mode. The prompt and local validation still require one JSON object with the six specified fields. [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) · [DeepSeek JSON Output](https://api-docs.deepseek.com/guides/json_mode/) · [Qwen OpenAI Compatibility](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)

The runner saves each attempt to `outputs/results.jsonl`. Rerun with the same settings to skip valid responses and retry failed or unattempted inputs. Use a distinct `--output` for each model, dataset, or JSON mode and pass it to `3_analyze_results.py --results`. Run one evaluator per results path.

The selected endpoint receives the vignette text. Keep the source tables, derived cases, reports, and credentials local to your working copy. Git ignores `data/`, `outputs/`, and `.env` files.

### MedGemma on Colab

Use a CUDA runtime with PyTorch, Transformers 4.56 or newer, and Accelerate. Mount your private Drive and place `2_run_eval.py`, `3_analyze_results.py`, and the generated `outputs/cases.jsonl` in `MyDrive/mimic-llm-eval/`. The example assumes the MedGemma 1.5 4B IT checkpoint was copied to `/content/medgemma-local`, with `config.json` directly inside it. The runner uses greedy decoding with up to 512 generated tokens. It uses BF16 on GPUs with native BF16 support and FP32 on GPUs such as the T4. FP32 uses more memory and may offload layers to the CPU. Use a separate results path when changing precision. [Google MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card) · [Accelerate device placement](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)

Run this cell first to attempt the three variants of the first admission.

```python
%cd /content/drive/MyDrive/mimic-llm-eval
!python 2_run_eval.py --local-model /content/medgemma-local --output outputs/medgemma_results.jsonl --limit 3
```

Check that all three responses score successfully. The local runner accepts one complete Markdown `json` code block around the response and validates the object with the same six-field schema. It preserves the original text for wrapped or invalid responses. On resume, the runner rechecks saved text before making another model call. Then run the full evaluation and analysis in a separate cell. The results path is on Drive, so the runner can resume after a disconnect when the checkpoint and runtime settings match.

```python
!python 2_run_eval.py --local-model /content/medgemma-local --output outputs/medgemma_results.jsonl
!python 3_analyze_results.py --results outputs/medgemma_results.jsonl
```

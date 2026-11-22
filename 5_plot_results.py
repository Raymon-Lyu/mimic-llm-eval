"""从 v2 评测结果生成简洁的出版级 2 联汇总图（assets/v2_benchmark_overview.png）。"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    ("deepseek-v4-pro-0813", "DeepSeek\nV4-Pro", "DeepSeek-V4-Pro"),
    ("deepseek-v4.1-flash-1017", "DeepSeek\nV4.1-Flash", "DeepSeek-V4.1-Flash"),
    ("qwen3.8-flash", "Qwen3.8\nFlash", "Qwen3.8-Flash"),
    ("qwen4-max", "Qwen4\nMax", "Qwen4-Max"),
    ("medgemma-local", "MedGemma-4B\n(NF4)", "MedGemma-4B (NF4)"),
]

MODEL_COLORS = {
    "deepseek-v4-pro-0813": "#1D4ED8",
    "deepseek-v4.1-flash-1017": "#60A5FA",
    "qwen3.8-flash": "#0D9488",
    "qwen4-max": "#7C3AED",
    "medgemma-local": "#DC2626",
}

CONDITIONS = ("full", "lab_only", "key_masked", "targeted_conflict")
COND_LABELS = ["Full\n(Vitals+Labs)", "Lab-Only\n(Ablation)", "Key-Masked\n(−Vitals, −50%)", "Targeted\nConflict"]


def load_v2_stats(cases_path, results_dir):
    case_bytes = cases_path.read_bytes()
    expected_sha = hashlib.sha256(case_bytes).hexdigest()
    stats = {}
    for path in sorted(results_dir.glob("results_*.jsonl")):
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        run = lines[0].get("run", {})
        if run.get("cases_sha256") != expected_sha:
            continue
        model_id = run.get("model", path.stem)
        by_cond = {c: [] for c in CONDITIONS}
        for row in lines[1:]:
            if "response" in row and row["condition"] in by_cond:
                by_cond[row["condition"]].append(row["response"])
        stats[model_id] = {
            "full_diag": 100.0 * sum(r["action"] == "diagnose" for r in by_cond["full"]) / max(len(by_cond["full"]), 1),
            "key_masked_diag": 100.0 * sum(r["action"] == "diagnose" for r in by_cond["key_masked"]) / max(len(by_cond["key_masked"]), 1),
            "conflict_defer": 100.0 * sum(r["action"] == "defer_conflict" for r in by_cond["targeted_conflict"]) / max(len(by_cond["targeted_conflict"]), 1),
            "conf_curve": [mean(r["confidence"] for r in by_cond[c]) if by_cond[c] else 0.0 for c in CONDITIONS],
            "scored": sum(len(v) for v in by_cond.values()),
        }
    return stats


def make_figure(stats, output_path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    present = [(mid, short_lbl, full_lbl) for mid, short_lbl, full_lbl in MODEL_ORDER if mid in stats]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.5), dpi=300, gridspec_kw={"width_ratios": [1.18, 1.0]})
    fig.patch.set_facecolor("white")

    # --- Panel A: Key Reliability Rates (%) ---
    x = np.arange(len(present))
    w = 0.25
    full_vals = [stats[m]["full_diag"] for m, _, _ in present]
    mask_vals = [stats[m]["key_masked_diag"] for m, _, _ in present]
    conf_vals = [stats[m]["conflict_defer"] for m, _, _ in present]

    b1 = ax1.bar(x - w, full_vals, width=w, color="#475569", label="Full: Direct Diagnose (%)")
    b2 = ax1.bar(x, mask_vals, width=w, color="#DC2626", label="Key-Masked: Premature Diagnose ↓ (%)")
    b3 = ax1.bar(x + w, conf_vals, width=w, color="#0D9488", label="Targeted Conflict: Defer Conflict ↑ (%)")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, h + 1.8,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=8.3, color="#1E293B",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([short_lbl for _, short_lbl, _ in present], fontsize=9.0)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylim(0, 130)
    ax1.set_ylabel("Rate (%)", fontsize=10, color="#0F172A")
    ax1.set_title("A. Clinical Action Behavior Across Perturbations", fontsize=11.5, fontweight="bold", pad=12, loc="left", color="#0F172A")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.01))

    # --- Panel B: Mean Confidence Trajectory ---
    cx = np.arange(len(CONDITIONS))
    for mid, _, full_lbl in present:
        y = stats[mid]["conf_curve"]
        lw = 2.6 if mid in {"deepseek-v4-pro-0813", "medgemma-local"} else 2.0
        ls = "--" if mid == "medgemma-local" else "-"
        ax2.plot(
            cx, y, marker="o", markersize=6, linewidth=lw, linestyle=ls,
            color=MODEL_COLORS[mid], label=full_lbl,
        )
        # Annotate start and end values cleanly
        offset_end = {
            "deepseek-v4-pro-0813": 1.8,
            "deepseek-v4.1-flash-1017": -1.0,
            "qwen3.8-flash": -2.5,
            "qwen4-max": 1.5,
            "medgemma-local": 1.5,
        }.get(mid, 0.0)
        ax2.text(
            cx[-1] + 0.08, y[-1] + offset_end, f"{y[-1]:.1f}",
            va="center", ha="left", fontsize=8.5, fontweight="bold", color=MODEL_COLORS[mid],
        )

    ax2.set_xticks(cx)
    ax2.set_xticklabels(COND_LABELS, fontsize=8.8)
    ax2.set_xlim(-0.2, 3.42)
    ax2.set_ylim(-3, 96)
    ax2.set_ylabel("Mean Self-Reported Confidence (0–100)", fontsize=10, color="#0F172A")
    ax2.set_title("B. Confidence Calibration Across Conditions", fontsize=11.5, fontweight="bold", pad=12, loc="left", color="#0F172A")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.35, color="#94A3B8")
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=8.3, loc="upper right")

    fig.suptitle(
        "MIMIC-IV Clinical LLM Reliability Pilot (v2 · 20 Severe Distinct Patients × 4 Paired Conditions)",
        fontsize=12.5, fontweight="bold", y=0.99, color="#0F172A",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("outputs/cases_v2.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/v2"))
    parser.add_argument("--output", type=Path, default=Path("assets/v2_benchmark_overview.png"))
    args = parser.parse_args()
    stats = load_v2_stats(args.cases, args.results_dir)
    make_figure(stats, args.output)


if __name__ == "__main__":
    main()

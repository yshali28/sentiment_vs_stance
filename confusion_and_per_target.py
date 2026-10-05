"""
confusion_and_per_target.py

Run this AFTER compare_phase1_phase2.py — it reads the merged CSV that
script already saved, so there's no retraining and no re-downloading here.
Produces two figures and one CSV for the report:
  outputs/confusion_matrices.png     - Phase 1 vs Phase 2 confusion matrices
  outputs/per_target_accuracy.png    - bar chart of accuracy by target
  outputs/per_target_breakdown.csv   - the numbers behind that chart

Run from your project root:
    python confusion_and_per_target.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

MERGED_PATH = "outputs/phase1_vs_phase2_merged.csv"
STANCE_LABELS = ["FAVOR", "AGAINST", "NONE"]

df = pd.read_csv(MERGED_PATH)
print(f"Loaded {len(df)} rows from {MERGED_PATH}")

# ---------------------------------------------------------------------
# Part 1: confusion matrices, Phase 1 vs Phase 2, side by side
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
cols = ["pred_stance_phase1", "pred_stance_phase2"]
titles = ["Phase 1 (TF-IDF + LogReg)", "Phase 2 (Target-Aware BERTweet)"]
cmaps = ["Greys", "Reds"]

for ax, col, title, cmap in zip(axes, cols, titles, cmaps):
    cm = confusion_matrix(df["stance"], df[col], labels=STANCE_LABELS)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=STANCE_LABELS, yticklabels=STANCE_LABELS, cbar=False)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (gold)")

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)
print("Saved outputs/confusion_matrices.png")

# quick printout of the diagonal (correct predictions) for both, since
# that's the number worth quoting in the report
for col, title in zip(cols, titles):
    cm = confusion_matrix(df["stance"], df[col], labels=STANCE_LABELS)
    diag = dict(zip(STANCE_LABELS, cm.diagonal()))
    print(f"{title} — correct predictions per class: {diag}")

# ---------------------------------------------------------------------
# Part 2: per-target breakdown
# ---------------------------------------------------------------------
rows = []
for target, g in df.groupby("target"):
    acc1 = (g["pred_stance_phase1"] == g["stance"]).mean()
    acc2 = (g["pred_stance_phase2"] == g["stance"]).mean()

    # same divergent-tweet definition as disentangle.py
    divergent = ((g["stance"] == "FAVOR") & (g["sentiment"] == "neg")) | \
                ((g["stance"] == "AGAINST") & (g["sentiment"] == "pos"))
    div_rate = divergent.mean()
    div_g = g[divergent]

    acc1_div = (div_g["pred_stance_phase1"] == div_g["stance"]).mean() if len(div_g) else float("nan")
    acc2_div = (div_g["pred_stance_phase2"] == div_g["stance"]).mean() if len(div_g) else float("nan")

    rows.append({
        "target": target,
        "n": len(g),
        "divergence_rate_%": round(div_rate * 100, 1),
        "phase1_acc": round(acc1, 3),
        "phase2_acc": round(acc2, 3),
        "delta": round(acc2 - acc1, 3),
        "n_divergent": len(div_g),
        "phase1_acc_divergent": round(acc1_div, 3),
        "phase2_acc_divergent": round(acc2_div, 3),
    })

summary = pd.DataFrame(rows).sort_values("divergence_rate_%", ascending=False)
print("\nPer-target breakdown:")
print(summary.to_string(index=False))
summary.to_csv("outputs/per_target_breakdown.csv", index=False)

# ---------------------------------------------------------------------
# Part 3: per-target accuracy chart
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.8))
x = range(len(summary))
w = 0.35
ax.bar([i - w / 2 for i in x], summary["phase1_acc"], width=w, label="Phase 1", color="#8A8A8A")
ax.bar([i + w / 2 for i in x], summary["phase2_acc"], width=w, label="Phase 2", color="#7A1F2B")
ax.set_xticks(list(x))
ax.set_xticklabels(summary["target"], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Stance accuracy")
ax.set_title("Per-Target Stance Accuracy — Phase 1 vs Phase 2")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/per_target_accuracy.png", dpi=150)
print("\nSaved outputs/per_target_accuracy.png")
print("Saved outputs/per_target_breakdown.csv")

"""
compare_phase1_phase2.py

Compares your already-saved Phase 1 and Phase 2 predictions — no retraining,
just reads two CSVs and analyses them. Run this from your project root
(the folder containing data/, src/, outputs/).

Needs:
  outputs/phase1_predictions.csv   <- from running src/export_predictions.py
  phase2_predictions.csv           <- downloaded from your Kaggle/Colab run
                                       (put it in the project root, or edit
                                       PHASE2_PATH below to point at it)
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

PHASE1_PATH = "outputs/phase1_predictions.csv"
PHASE2_PATH = "phase2_predictions.csv"

# ---------------------------------------------------------------------
# Step 1: load both prediction files
# ---------------------------------------------------------------------
phase1 = pd.read_csv(PHASE1_PATH)
phase2 = pd.read_csv(PHASE2_PATH)
print(f"Phase 1 predictions: {len(phase1)} rows")
print(f"Phase 2 predictions: {len(phase2)} rows")

# ---------------------------------------------------------------------
# Step 2: merge on (text, target) — safer than assuming row order matches,
# since the two files were generated in different environments
# ---------------------------------------------------------------------
merged = phase1.merge(
    phase2[["text", "target", "pred_stance", "pred_sentiment"]],
    on=["text", "target"],
    suffixes=("_phase1", "_phase2"),
)
print(f"Merged rows: {len(merged)}")

# if your phase1 file's prediction column is just "pred_stance", the merge
# above will rename both to pred_stance_phase1 / pred_stance_phase2 — handle
# the case where phase1's column didn't need renaming (no clash)
p1_col = "pred_stance_phase1" if "pred_stance_phase1" in merged.columns else "pred_stance"
p2_col = "pred_stance_phase2" if "pred_stance_phase2" in merged.columns else "pred_stance"

# ---------------------------------------------------------------------
# Step 3: official macro-F1 for both
# ---------------------------------------------------------------------
f1_phase1 = f1_score(merged["stance"], merged[p1_col], labels=["FAVOR", "AGAINST"], average="macro", zero_division=0)
f1_phase2 = f1_score(merged["stance"], merged[p2_col], labels=["FAVOR", "AGAINST"], average="macro", zero_division=0)
print(f"\nPhase 1 official stance macro-F1: {f1_phase1:.4f}")
print(f"Phase 2 official stance macro-F1: {f1_phase2:.4f}")
print(f"Improvement: {f1_phase2 - f1_phase1:+.4f}")

# ---------------------------------------------------------------------
# Step 4: the divergent subset — same definition as disentangle.py
# ---------------------------------------------------------------------
divergent = ((merged["stance"] == "FAVOR") & (merged["sentiment"] == "neg")) | \
            ((merged["stance"] == "AGAINST") & (merged["sentiment"] == "pos"))
div = merged[divergent]
print(f"\nDivergent subset: {len(div)} / {len(merged)} tweets ({divergent.mean()*100:.1f}%)")

acc_p1_div = (div[p1_col] == div["stance"]).mean()
acc_p2_div = (div[p2_col] == div["stance"]).mean()
acc_p1_all = (merged[p1_col] == merged["stance"]).mean()
acc_p2_all = (merged[p2_col] == merged["stance"]).mean()

print(f"\nOverall stance accuracy      — Phase 1: {acc_p1_all:.4f}   Phase 2: {acc_p2_all:.4f}")
print(f"Divergent-subset accuracy    — Phase 1: {acc_p1_div:.4f}   Phase 2: {acc_p2_div:.4f}")

recovered = div[(div[p1_col] != div["stance"]) & (div[p2_col] == div["stance"])]
still_wrong = div[(div[p1_col] != div["stance"]) & (div[p2_col] != div["stance"])]
newly_broken = div[(div[p1_col] == div["stance"]) & (div[p2_col] != div["stance"])]

print(f"\nDivergent tweets Phase 1 got WRONG, Phase 2 got RIGHT (recovered): {len(recovered)}")
print(f"Divergent tweets BOTH got wrong: {len(still_wrong)}")
print(f"Divergent tweets Phase 1 had right, Phase 2 broke: {len(newly_broken)}")

if len(recovered) > 0:
    print("\nA few recovered examples (for your demo slide):")
    pd.set_option("display.max_colwidth", 100)
    print(recovered[["text", "target", "stance", "sentiment", p1_col, p2_col]].head(5).to_string(index=False))

# ---------------------------------------------------------------------
# Step 5: save merged CSV + comparison chart
# ---------------------------------------------------------------------
merged.to_csv("outputs/phase1_vs_phase2_merged.csv", index=False)

labels = ["Overall\nstance accuracy", "Accuracy on\ndivergent subset", "Official macro-F1\n(FAVOR/AGAINST)"]
phase1_vals = [acc_p1_all, acc_p1_div, f1_phase1]
phase2_vals = [acc_p2_all, acc_p2_div, f1_phase2]

x = range(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.bar([i - w/2 for i in x], phase1_vals, width=w, label="Phase 1 (TF-IDF + LogReg)", color="#8A8A8A")
ax.bar([i + w/2 for i in x], phase2_vals, width=w, label="Phase 2 (Target-Aware BERTweet)", color="#7A1F2B")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylim(0, max(phase2_vals) + 0.15)
ax.set_ylabel("Score")
ax.set_title("Phase 1 vs Phase 2 — Stance Detection Performance")
for i, (a, b) in enumerate(zip(phase1_vals, phase2_vals)):
    ax.text(i - w/2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w/2, b + 0.015, f"{b:.3f}", ha="center", fontsize=9)
ax.legend(loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/phase1_vs_phase2.png", dpi=150)

print("\nSaved: outputs/phase1_vs_phase2_merged.csv, outputs/phase1_vs_phase2.png")

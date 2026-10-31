#!/usr/bin/env python
"""Make all report figures from results/results.json.   python make_figures.py [results_dir]"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rd = sys.argv[1] if len(sys.argv) > 1 else "results"
R = json.load(open(os.path.join(rd, "results.json")))
out = os.path.join(rd, "figures")
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 8, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


# Q1a: tree accuracy vs depth / leaf size
g = R["q1a"]["grid"]
depths = [str(d) for d in dict.fromkeys(r["max_depth"] for r in g)]
leaves = sorted({r["min_samples_leaf"] for r in g})
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
for a, key in zip(ax, ("train", "val")):
    for leaf in leaves:
        a.plot(depths, [r[key] for r in g if r["min_samples_leaf"] == leaf], "o-", label=f"leaf={leaf}")
    a.set_title(f"Single tree: {key} accuracy"); a.set_xlabel("max depth"); a.set_ylabel("accuracy"); a.legend()
save(fig, "q1a_tree_grid.pdf")

# Q1b: bagging accuracy vs B
rows = R["q1bc"]["rows"]
B = [r["B"] for r in rows]
fig, a = plt.subplots(figsize=(3.2, 2.4))
for k in ("train", "val", "test"):
    a.plot(B, [100 * r[k] for r in rows], "o-", label=k)
a.set_xscale("log"); a.set_xlabel("number of trees B"); a.set_ylabel("accuracy (%)"); a.legend()
save(fig, "q1b_bagging_accuracy.pdf")

# Q1c: OOB, disagreement, margin
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2))
for a, key, t in zip(ax, ("oob_acc", "disagreement_val", "margin_val"),
                     ("OOB accuracy", "pairwise disagreement (val)", "mean vote margin (val)")):
    xs = [r["B"] for r in rows if r[key] is not None]
    a.plot(xs, [r[key] for r in rows if r[key] is not None], "o-")
    a.axvline(R["q1bc"]["B_final"], ls=":", c="gray")
    a.set_xscale("log"); a.set_xlabel("B"); a.set_title(t)
save(fig, "q1c_diagnostics.pdf")

# Q2b: SAMME grid + diagnostics of selected model
bg = R["q2b"]["grid"]
ds = sorted({r["depth"] for r in bg})
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.4))
for a, key in zip(ax, ("val", "train")):
    for d in ds:
        a.plot([r["rounds"] for r in bg if r["depth"] == d], [r[key] for r in bg if r["depth"] == d], "o-", label=f"depth {d}")
    a.set_title(f"SAMME {key} accuracy"); a.set_xlabel("rounds"); a.legend()
save(fig, "q2b_samme_grid.pdf")

dg = R["q2b"]["diagnostics"]
fig, ax = plt.subplots(1, 4, figsize=(8, 2.1))
for a, key, t in zip(ax, ("weighted_error", "alpha", "max_sample_weight", "top10_weight"),
                     ("weighted train error", "learner weight $\\alpha_t$", "max sample weight", "weight of 10 heaviest")):
    a.plot(range(1, len(dg[key]) + 1), dg[key]); a.set_title(t); a.set_xlabel("round")
save(fig, "q2b_diagnostics.pdf")

# Q2c: heavy-sample history
hs = R["q2c"]["heavy_sample"]
fig, a = plt.subplots(figsize=(3.4, 2.4))
a.plot(hs["weight_history"], label="sample weight")
miss = [i for i, c in enumerate(hs["correct_history"]) if not c]
a.scatter(miss, [hs["weight_history"][i] for i in miss], c="r", s=8, label="misclassified in that round")
a.set_xlabel("round"); a.set_ylabel("weight"); a.legend()
save(fig, "q2c_heavy_sample.pdf")

# Q3a: bias/variance bars
m = ["tree", "bagging", "boosting"]
q = R["q3a"]
x = np.arange(3)
fig, a = plt.subplots(figsize=(3.2, 2.4))
a.bar(x - 0.2, [q[k]["bias2"] for k in m], 0.4, label="bias$^2$")
a.bar(x + 0.2, [q[k]["variance"] for k in m], 0.4, label="variance")
a.set_xticks(x); a.set_xticklabels(m); a.legend()
save(fig, "q3a_bias_variance.pdf")

# Q3b: depth study
ds_ = R["q3bc"]["rows"]
lab = [str(r["max_depth"]) for r in ds_]
fig, ax = plt.subplots(1, 3, figsize=(7, 2.2))
for a, (k, t) in zip(ax, (("mean_acc", "mean test accuracy"), ("bias2", "bias$^2$"), ("variance", "variance"))):
    a.plot(lab, [r["tree"][k] for r in ds_], "o-", label="single tree")
    a.plot(lab, [r["bag"][k] for r in ds_], "s-", label="bagging")
    a.set_title(t); a.set_xlabel("max depth"); a.legend()
save(fig, "q3b_depth_study.pdf")

# Q3d: final comparison
t = R["q3d"]
fig, ax = plt.subplots(1, 2, figsize=(6, 2.3))
ax[0].bar(x - 0.2, [100 * t[k]["clean_acc"] for k in m], 0.4, label="clean")
ax[0].bar(x + 0.2, [100 * t[k]["masked_acc"] for k in m], 0.4, label="masked")
ax[0].set_title("test accuracy (%)")
ax[1].bar(x - 0.2, [t[k]["bias2"] for k in m], 0.4, label="bias$^2$")
ax[1].bar(x + 0.2, [t[k]["variance"] for k in m], 0.4, label="variance")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(m); a.legend()
save(fig, "q3d_final_comparison.pdf")
print("figures written to", out)

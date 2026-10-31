#!/usr/bin/env python
"""
Machine Learning Assignment 2: bagging, boosting and bias-variance on Olivetti faces.

Everything (single tree, bagging, SAMME, masking, 30-refit bias/variance) is run from here.
Results are written to  <out>/results.json  and the split to  <out>/data/saved_split.npz.
Figures are made by make_figures.py from results.json.

Usage:
    python run_experiments.py                # full run (takes a while, see README)
    python run_experiments.py --smoke        # tiny synthetic run, only to test the code
"""
import argparse
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

ROLL_NO = "M26ECM505"
SEED = int("".join(ch for ch in ROLL_NO if ch.isdigit()))  # all numeric characters, in order -> 26505
K = 40                      # identities
MASK_LO, MASK_HI = 24, 40   # rows/cols 24..39 inclusive -> centred 16x16 block

FULL = dict(
    tree_depths=[2, 4, 8, 16, None], tree_leaves=[1, 2, 4],
    B_list=[1, 5, 10, 25, 50, 100, 200],
    boost_depths=[1, 2, 3, 4], boost_rounds=[10, 25, 50, 100],
    compare_L=[25, 50, 100],
    R=30, depth_study_B=100,
)
SMOKE = dict(
    tree_depths=[2, None], tree_leaves=[1, 4],
    B_list=[1, 5, 10],
    boost_depths=[1, 2], boost_rounds=[2, 4],
    compare_L=[2, 4],
    R=2, depth_study_B=5,
)


# ----------------------------------------------------------------------------- data
def load_data(smoke):
    if smoke:  # synthetic stand-in, ONLY for testing the pipeline
        rng = np.random.default_rng(0)
        y = np.repeat(np.arange(K), 10)
        protos = rng.normal(size=(K, 4096))
        X = protos[y] + 1.5 * rng.normal(size=(400, 4096))
        return X.astype(np.float32), y
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces()
    return d.data.astype(np.float32), d.target  # raw flattened pixels, shape (400, 4096)


def get_split(y, path):
    """6 train / 2 val / 2 test per identity, seeded by the roll number. Saved once, reused everywhere."""
    if os.path.exists(path):
        s = np.load(path)
        return {k: s[k] for k in ("train", "val", "test")}
    rng = np.random.default_rng(SEED)
    tr, va, te = [], [], []
    for c in range(K):
        idx = rng.permutation(np.where(y == c)[0])
        tr += list(idx[:6]); va += list(idx[6:8]); te += list(idx[8:10])
    split = dict(train=np.array(tr), val=np.array(va), test=np.array(te))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **split)
    return split


def mask_images(X):
    Xm = X.reshape(-1, 64, 64).copy()
    Xm[:, MASK_LO:MASK_HI, MASK_LO:MASK_HI] = 0.0
    return Xm.reshape(len(X), -1)


def acc(pred, y):
    return float((pred == y).mean())


def tree_stats(tree, Xtr):
    leaf_ids = tree.apply(Xtr)
    _, counts = np.unique(leaf_ids, return_counts=True)
    n_leaves = int(tree.get_n_leaves())
    return dict(n_leaves=n_leaves,
                mean_samples_per_leaf=float(len(Xtr) / n_leaves),
                frac_singleton_leaves=float((counts == 1).sum() / n_leaves),
                depth_reached=int(tree.get_depth()))


# ----------------------------------------------------------------------------- bagging (from scratch)
def votes(P):
    """P: (B, N) predicted labels -> (N, K) vote counts."""
    N = P.shape[1]
    C = np.zeros((N, K))
    ar = np.arange(N)
    for p in P:
        C[ar, p] += 1
    return C


def vote_pred(C):
    return C.argmax(1)  # ties -> lowest class index (deterministic)


def disagreement_per_sample(C, B):
    """Mean over tree pairs of 1[pred_i != pred_j], computed from vote counts."""
    if B < 2:
        return np.full(len(C), np.nan)
    agree = (C * (C - 1)).sum(1) / (B * (B - 1))
    return 1.0 - agree


def margin_per_sample(C, B):
    s = np.sort(C / B, axis=1)
    return s[:, -1] - s[:, -2]


class Bagging:
    def __init__(self, n_trees, tree_kw, seed, n_jobs=-1):
        self.n_trees, self.tree_kw, self.seed, self.n_jobs = n_trees, tree_kw, seed, n_jobs

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.n_train = n
        # bootstrap indices and tree seeds are drawn sequentially, so results do not depend on n_jobs
        self.boots = [rng.integers(0, n, n) for _ in range(self.n_trees)]   # size n, with replacement
        seeds = [int(rng.integers(2**31 - 1)) for _ in range(self.n_trees)]

        def fit_one(idx, s):
            return DecisionTreeClassifier(**self.tree_kw, random_state=s).fit(X[idx], y[idx])

        self.trees = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(fit_one)(idx, s) for idx, s in zip(self.boots, seeds))
        return self

    def pred_matrix(self, X):
        return np.stack([t.predict(X) for t in self.trees])  # (B, N)

    def votes(self, X, B=None, P=None):
        P = self.pred_matrix(X) if P is None else P
        return votes(P[: (B or len(self.trees))])

    def oob(self, X, y, B):
        n = len(X)
        C = np.zeros((n, K))
        for t, idx in zip(self.trees[:B], self.boots[:B]):
            oob_idx = np.setdiff1d(np.arange(n), idx)
            if len(oob_idx):
                C[oob_idx, t.predict(X[oob_idx])] += 1
        cov = C.sum(1) > 0
        return (float((C.argmax(1)[cov] == y[cov]).mean()) if cov.any() else float("nan"),
                float(cov.mean()))

    def mean_distinct(self, B):
        return float(np.mean([len(np.unique(b)) for b in self.boots[:B]]))


# ----------------------------------------------------------------------------- SAMME (from scratch)
def samme_round(w, miss, K_):
    """One SAMME weight update. w sums to 1, miss is a boolean array. Returns (eps, alpha, new_w)."""
    eps = float(w[miss].sum() / w.sum())
    if eps >= 1 - 1.0 / K_:
        return eps, None, w                      # not better than random guessing: caller must handle
    eps_c = max(eps, 1e-12)                       # avoid log(0) for a perfect learner
    alpha = float(np.log((1 - eps_c) / eps_c) + np.log(K_ - 1))
    new_w = w * np.exp(alpha * miss)
    return eps, alpha, new_w / new_w.sum()


class SAMME:
    def __init__(self, depth, rounds, seed):
        self.depth, self.rounds, self.seed = depth, rounds, seed

    def fit(self, X, y):
        n = len(X)
        w = np.full(n, 1.0 / n)
        rng = np.random.default_rng(self.seed)
        self.learners, self.alphas = [], []
        self.hist = dict(eps=[], alpha=[], max_w=[], top10_w=[], w=[], correct=[])
        self.stop_reason = "completed"
        for t in range(self.rounds):
            h = DecisionTreeClassifier(max_depth=self.depth, random_state=int(rng.integers(2**31 - 1)))
            h.fit(X, y, sample_weight=w)
            miss = h.predict(X) != y
            eps, alpha, w_new = samme_round(w, miss, K)
            if alpha is None:   # weak learner no better than random multiclass guessing -> discard and stop
                self.stop_reason = f"round {t}: weighted error {eps:.4f} >= 1-1/K, learner discarded, boosting stopped"
                break
            self.learners.append(h)
            self.alphas.append(alpha)
            self.hist["eps"].append(eps)
            self.hist["alpha"].append(alpha)
            self.hist["max_w"].append(float(w.max()))
            self.hist["top10_w"].append(float(np.sort(w)[-10:].sum()))
            self.hist["w"].append(w.copy())
            self.hist["correct"].append(~miss)
            w = w_new
            if eps <= 1e-12:    # perfect learner: weights would not change, stop
                self.stop_reason = f"round {t}: zero weighted error, stopped"
                break
        return self

    def pred_matrix(self, X):
        return np.stack([h.predict(X) for h in self.learners])

    def scores(self, X, T=None, P=None):
        T = T or len(self.learners)
        P = self.pred_matrix(X) if P is None else P
        S = np.zeros((P.shape[1], K))
        ar = np.arange(P.shape[1])
        for t in range(T):
            S[ar, P[t]] += self.alphas[t]
        return S

    def proba(self, X, T=None, P=None):
        T = T or len(self.learners)
        return self.scores(X, T, P) / sum(self.alphas[:T])

    def predict(self, X, T=None, P=None):
        return self.scores(X, T, P).argmax(1)


def toy_example():
    """Q2a: 3-class, 10-sample example; logs one complete SAMME round."""
    Xt = np.arange(10, dtype=float).reshape(-1, 1)
    yt = np.array([0, 0, 0, 1, 1, 0, 1, 2, 2, 1])
    w0 = np.full(10, 0.1)
    h = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xt, yt, sample_weight=w0)
    pred = h.predict(Xt)
    eps, alpha, w1 = samme_round(w0, pred != yt, 3)
    log = dict(X=Xt.ravel().tolist(), y=yt.tolist(), K=3, initial_weights=w0.tolist(),
               weak_learner_predictions=pred.tolist(), misclassified=(pred != yt).tolist(),
               weighted_error=eps, alpha=alpha, updated_normalised_weights=w1.tolist(),
               weights_sum=float(w1.sum()))
    return log


# ----------------------------------------------------------------------------- bias / variance
def bias_var(P, y_true):
    """P: (R, N, K) class vectors from R refits; y_true: (N,)."""
    E = np.eye(K)[y_true]
    pbar = P.mean(0)
    bias2 = float(((pbar - E) ** 2).sum(1).mean())
    var = float(((P - pbar) ** 2).sum(2).mean())
    return bias2, var


def resample_sets(split, y, R):
    """R training sets: within each identity, 6 draws with replacement from that identity's 6 train faces."""
    rng = np.random.default_rng([SEED, 3])
    tr = split["train"]
    groups = [tr[y[tr] == c] for c in range(K)]
    return [np.concatenate([rng.choice(g, 6, replace=True) for g in groups]) for _ in range(R)]


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run to test the pipeline")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = "results_smoke" if args.smoke else "results"
    cfg = SMOKE if args.smoke else FULL
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    X, y = load_data(args.smoke)
    split_path = (os.path.join(args.out, "data", "saved_split.npz") if args.smoke
                  else os.path.join("data", "saved_split.npz"))
    split = get_split(y, split_path)
    Xtr, ytr = X[split["train"]], y[split["train"]]
    Xva, yva = X[split["val"]], y[split["val"]]
    Xte, yte = X[split["test"]], y[split["test"]]
    Xte_m = mask_images(Xte)
    log(f"seed={SEED}  train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")

    R = dict(meta=dict(roll_no=ROLL_NO, seed=SEED, smoke=args.smoke, config=cfg))

    # ================================================================= Q1a single tree
    grid = []
    for d in cfg["tree_depths"]:
        for leaf in cfg["tree_leaves"]:
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=SEED).fit(Xtr, ytr)
            grid.append(dict(max_depth=d, min_samples_leaf=leaf,
                             train=acc(t.predict(Xtr), ytr), val=acc(t.predict(Xva), yva),
                             test=acc(t.predict(Xte), yte), **tree_stats(t, Xtr)))
    # selection uses validation accuracy ONLY (ties -> fewer leaves, i.e. simpler tree)
    best = max(grid, key=lambda r: (r["val"], -r["n_leaves"]))
    tree_kw = dict(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"])
    tree = DecisionTreeClassifier(**tree_kw, random_state=SEED).fit(Xtr, ytr)
    R["q1a"] = dict(grid=grid, selected=best, tree_kw=tree_kw)
    log(f"[Q1a] selected {tree_kw}: val={best['val']:.4f}")

    # ================================================================= Q1b/c bagging
    Bmax = max(cfg["B_list"])
    bag = Bagging(Bmax, tree_kw, SEED).fit(Xtr, ytr)    # nested ensembles: first B trees of one big bag
    P = {k: bag.pred_matrix(Xk) for k, Xk in dict(train=Xtr, val=Xva, test=Xte, test_masked=Xte_m).items()}
    n = len(Xtr)
    theory = n * (1 - (1 - 1.0 / n) ** n)
    rows = []
    for B in cfg["B_list"]:
        r = dict(B=B, mean_distinct=bag.mean_distinct(B), theory_distinct=theory)
        for k in ("train", "val", "test"):
            r[k] = acc(vote_pred(votes(P[k][:B])), dict(train=ytr, val=yva, test=yte)[k])
        r["oob_acc"], r["oob_coverage"] = bag.oob(Xtr, ytr, B)
        Cv = votes(P["val"][:B])
        r["disagreement_val"] = float(disagreement_per_sample(Cv, B).mean()) if B > 1 else None
        r["margin_val"] = float(margin_per_sample(Cv, B).mean())
        rows.append(r)
        log(f"[Q1b/c] B={B:3d} train={r['train']:.4f} val={r['val']:.4f} test={r['test']:.4f} "
            f"oob={r['oob_acc']:.4f} dis={r['disagreement_val']} margin={r['margin_val']:.4f}")
    Bf = max(rows, key=lambda r: (r["val"], -r["B"]))["B"]   # validation only, ties -> smaller B
    R["q1bc"] = dict(rows=rows, B_final=Bf)
    log(f"[Q1c] final B = {Bf}")

    # ================================================================= Q1d clean vs masked
    def tree_eval(Xk, yk):
        pr = tree.predict_proba(Xk)
        return pr.argmax(1), pr.max(1)

    def bag_eval(Pk):
        C = votes(Pk[:Bf])
        return C.argmax(1), C.max(1) / Bf, C

    tp, tconf = tree_eval(Xte, yte)
    tpm, tconf_m = tree_eval(Xte_m, yte)
    bp, bconf, Cte = bag_eval(P["test"])
    bpm, bconf_m, Cte_m = bag_eval(P["test_masked"])
    tree_classes = tree.classes_
    tp, tpm = tree_classes[tp], tree_classes[tpm]

    def case(i):
        C = Cte[i]
        top = np.argsort(-C)[:5]
        return dict(test_pos=int(i), dataset_index=int(split["test"][i]), true=int(yte[i]),
                    tree_pred=int(tp[i]), bag_pred=int(bp[i]),
                    bag_votes={int(k): int(C[k]) for k in top if C[k] > 0})

    fixed = np.where((tp != yte) & (bp == yte))[0][:2]
    both = np.where((tp != yte) & (bp != yte))[0][:2]
    R["q1d"] = dict(
        tree=dict(clean_acc=acc(tp, yte), masked_acc=acc(tpm, yte),
                  clean_conf=float(tconf.mean()), masked_conf=float(tconf_m.mean())),
        bagging=dict(B=Bf, clean_acc=acc(bp, yte), masked_acc=acc(bpm, yte),
                     clean_conf=float(bconf.mean()), masked_conf=float(bconf_m.mean())),
        bagging_corrects_tree=[case(i) for i in fixed],
        both_wrong=[case(i) for i in both])
    log(f"[Q1d] tree clean/masked {acc(tp, yte):.4f}/{acc(tpm, yte):.4f}   "
        f"bag clean/masked {acc(bp, yte):.4f}/{acc(bpm, yte):.4f}")

    # ================================================================= Q2a toy SAMME round
    R["q2a"] = toy_example()
    log(f"[Q2a] toy round: eps={R['q2a']['weighted_error']:.4f} alpha={R['q2a']['alpha']:.6f}")

    # ================================================================= Q2b grid
    max_rounds = max(cfg["boost_rounds"])
    bmodels, bgrid = {}, []
    for d in cfg["boost_depths"]:
        m = SAMME(d, max_rounds, SEED).fit(Xtr, ytr)
        bmodels[d] = m
        Ptr_, Pva_, Pte_ = m.pred_matrix(Xtr), m.pred_matrix(Xva), m.pred_matrix(Xte)
        for T in cfg["boost_rounds"]:    # prefixes of one sequential run == separate runs of T rounds
            Tu = min(T, len(m.learners))
            bgrid.append(dict(depth=d, rounds=T, rounds_used=Tu,
                              train=acc(m.predict(Xtr, Tu, Ptr_), ytr),
                              val=acc(m.predict(Xva, Tu, Pva_), yva),
                              test=acc(m.predict(Xte, Tu, Pte_), yte), stop_reason=m.stop_reason))
        log(f"[Q2b] depth={d}: {[(g['rounds'], round(g['val'], 4)) for g in bgrid if g['depth'] == d]}")
    bbest = max(bgrid, key=lambda r: (r["val"], -r["depth"], -r["rounds"]))   # validation only
    bd, bT = bbest["depth"], bbest["rounds_used"]
    boost = bmodels[bd]
    H = boost.hist
    R["q2b"] = dict(grid=bgrid, selected=bbest,
                    diagnostics=dict(weighted_error=H["eps"][:bT], alpha=H["alpha"][:bT],
                                     max_sample_weight=H["max_w"][:bT], top10_weight=H["top10_w"][:bT]))
    log(f"[Q2b] selected depth={bd} rounds={bT}  val={bbest['val']:.4f}")

    # ================================================================= Q2c boosting vs bagging
    cmp_rows = []
    for L in cfg["compare_L"]:
        t0 = time.perf_counter(); b_ = Bagging(L, tree_kw, SEED, n_jobs=1).fit(Xtr, ytr); tb = time.perf_counter() - t0
        t0 = time.perf_counter(); s_ = SAMME(bd, L, SEED).fit(Xtr, ytr); ts = time.perf_counter() - t0
        for name, mdl, tt in (("bagging", b_, tb), ("boosting", s_, ts)):
            if name == "bagging":
                pr = lambda Xk: vote_pred(mdl.votes(Xk))
            else:
                pr = lambda Xk: mdl.predict(Xk)
            cmp_rows.append(dict(method=name, learners=L, learners_used=len(getattr(mdl, "trees", None) or mdl.learners),
                                 train=acc(pr(Xtr), ytr), val=acc(pr(Xva), yva), test=acc(pr(Xte), yte),
                                 train_time_s=tt))
    # heavily weighted training face (selected boosting model, first bT rounds)
    W = np.stack(H["w"][:bT])                 # (T, n) weights used in each round
    corr = np.stack(H["correct"][:bT])        # (T, n) was the round-t learner right on that face?
    i_star = int(np.unravel_index(W.argmax(), W.shape)[1])
    t_star = int(W[:, i_star].argmax())
    final_pred_star = int(boost.predict(Xtr[i_star:i_star + 1], bT)[0])
    heavy = dict(train_pos=i_star, dataset_index=int(split["train"][i_star]), true=int(ytr[i_star]),
                 max_weight=float(W[:, i_star].max()), round_of_max=t_star,
                 weight_history=W[:, i_star].tolist(), correct_history=corr[:, i_star].tolist(),
                 final_ensemble_pred=final_pred_star,
                 frac_rounds_correct=float(corr[:, i_star].mean()))
    top10_final = np.argsort(-W[-1])[:10]
    R["q2c"] = dict(rows=cmp_rows, heavy_sample=heavy,
                    heaviest10_train_identities=[int(ytr[i]) for i in top10_final])
    for r in cmp_rows:
        log(f"[Q2c] {r['method']:8s} L={r['learners']:3d} train={r['train']:.4f} val={r['val']:.4f} "
            f"test={r['test']:.4f} time={r['train_time_s']:.1f}s")

    # ================================================================= Q2d frozen models, clean vs masked
    qc, qm = boost.proba(Xte, bT), boost.proba(Xte_m, bT)
    boc, bom = qc.argmax(1), qm.argmax(1)
    unchanged_bag = bp == bpm
    unchanged_boost = boc == bom
    diff_idx = np.where(unchanged_bag != unchanged_boost)[0][:3]
    Sc, Sm = boost.scores(Xte, bT), boost.scores(Xte_m, bT)

    def case2(i):
        top = lambda v, k=3: {int(j): round(float(v[j]), 4) for j in np.argsort(-v)[:k]}
        return dict(test_pos=int(i), dataset_index=int(split["test"][i]), true=int(yte[i]),
                    bag_clean_pred=int(bp[i]), bag_masked_pred=int(bpm[i]),
                    bag_clean_votes=top(Cte[i]), bag_masked_votes=top(Cte_m[i]),
                    tree_disagreement_clean=float(disagreement_per_sample(Cte[i:i + 1], Bf)[0]),
                    tree_disagreement_masked=float(disagreement_per_sample(Cte_m[i:i + 1], Bf)[0]),
                    boost_clean_pred=int(boc[i]), boost_masked_pred=int(bom[i]),
                    boost_clean_scores=top(qc[i]), boost_masked_scores=top(qm[i]))
    R["q2d"] = dict(
        tree=R["q1d"]["tree"], bagging=R["q1d"]["bagging"],
        boosting=dict(depth=bd, rounds=bT, clean_acc=acc(boc, yte), masked_acc=acc(bom, yte),
                      clean_conf=float(qc.max(1).mean()), masked_conf=float(qm.max(1).mean())),
        differing_response_cases=[case2(i) for i in diff_idx])
    log(f"[Q2d] boosting clean/masked {acc(boc, yte):.4f}/{acc(bom, yte):.4f}")

    # ================================================================= Q3a 30 refits
    Rn = cfg["R"]
    sets = resample_sets(split, y, Rn)
    PT, PB, PS = (np.zeros((Rn, len(yte), K)) for _ in range(3))
    accs = dict(tree=[], bagging=[], boosting=[])
    for r, idx in enumerate(sets):
        Xr, yr = X[idx], y[idx]
        t = DecisionTreeClassifier(**tree_kw, random_state=SEED + r + 1).fit(Xr, yr)
        PT[r] = np.eye(K)[t.predict(Xte)]
        b_ = Bagging(Bf, tree_kw, SEED + r + 1).fit(Xr, yr)
        PB[r] = b_.votes(Xte) / Bf
        s_ = SAMME(bd, bT, SEED + r + 1).fit(Xr, yr)
        PS[r] = s_.proba(Xte)
        for nm, Pm in (("tree", PT), ("bagging", PB), ("boosting", PS)):
            accs[nm].append(acc(Pm[r].argmax(1), yte))
        log(f"[Q3a] refit {r + 1}/{Rn}")
    q3a = {}
    for nm, Pm in (("tree", PT), ("bagging", PB), ("boosting", PS)):
        b2, v = bias_var(Pm, yte)
        q3a[nm] = dict(bias2=b2, variance=v, mean_acc=float(np.mean(accs[nm])), sd_acc=float(np.std(accs[nm], ddof=1)
                       if Rn > 1 else 0.0))
    R["q3a"] = q3a
    log(f"[Q3a] {json.dumps(q3a)}")

    # ================================================================= Q3b/c depth study
    sel_d = tree_kw["max_depth"]
    depths = [2, 16, None] if sel_d is None else [2, sel_d, None]   # see README: adaptation when selected is None
    DB = cfg["depth_study_B"]
    leaf = tree_kw["min_samples_leaf"]
    dstudy = []
    for d in depths:
        kw = dict(max_depth=d, min_samples_leaf=leaf)
        t0_ = DecisionTreeClassifier(**kw, random_state=SEED).fit(Xtr, ytr)
        orig_stats = tree_stats(t0_, Xtr)
        bag0 = Bagging(DB, kw, SEED).fit(Xtr, ytr)
        Cv0 = bag0.votes(Xva)
        dis = float(disagreement_per_sample(Cv0, DB).mean())
        PT2, PB2 = np.zeros((Rn, len(yte), K)), np.zeros((Rn, len(yte), K))
        a_t, a_b = [], []
        for r, idx in enumerate(sets):
            Xr, yr = X[idx], y[idx]
            PT2[r] = np.eye(K)[DecisionTreeClassifier(**kw, random_state=SEED + r + 1).fit(Xr, yr).predict(Xte)]
            PB2[r] = Bagging(DB, kw, SEED + r + 1).fit(Xr, yr).votes(Xte) / DB
            a_t.append(acc(PT2[r].argmax(1), yte)); a_b.append(acc(PB2[r].argmax(1), yte))
        bt, vt = bias_var(PT2, yte)
        bb, vb = bias_var(PB2, yte)
        dstudy.append(dict(max_depth=d, tree_stats=orig_stats,
                           tree=dict(bias2=bt, variance=vt, mean_acc=float(np.mean(a_t))),
                           bag=dict(bias2=bb, variance=vb, mean_acc=float(np.mean(a_b)), n_trees=DB),
                           variance_removed=vt - vb, variance_removed_frac=(vt - vb) / vt if vt else float("nan"),
                           acc_gain=float(np.mean(a_b) - np.mean(a_t)),
                           disagreement_val_original_bag=dis))
        log(f"[Q3b] depth={d}: tree var={vt:.4f} bag var={vb:.4f} gain={dstudy[-1]['acc_gain']:.4f}")
    gains = [r["acc_gain"] for r in dstudy]
    R["q3bc"] = dict(rows=dstudy, depth_with_max_gain=dstudy[int(np.argmax(gains))]["max_depth"],
                     deepest_is_max_gain=bool(np.argmax(gains) == len(dstudy) - 1))

    # ================================================================= Q3d final table
    R["q3d"] = {nm: dict(clean_acc=R["q2d"][key]["clean_acc"], masked_acc=R["q2d"][key]["masked_acc"],
                         bias2=q3a[nm]["bias2"], variance=q3a[nm]["variance"])
                for nm, key in (("tree", "tree"), ("bagging", "bagging"), ("boosting", "boosting"))}

    R["meta"]["runtime_s"] = time.time() - t_start
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log(f"done in {time.time() - t_start:.0f}s -> {args.out}/results.json")


if __name__ == "__main__":
    main()

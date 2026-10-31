# ML Assignment 2: Bagging, Boosting and Bias-Variance on Olivetti Faces

Name: K Gayatri Supraja | Roll No.: M26ECM505 | Seed: 26505 (all digits of the roll number, in order)
Repository: <paste link> | Final commit hash: <paste `git rev-parse HEAD` after your last commit>

## Files
- `run_experiments.py`: everything for Q1-Q3 (tree grid, bagging from scratch, SAMME from scratch,
  masking, 30 within-identity refits, depth study). Writes `results/results.json` and `data/saved_split.npz`.
- `make_figures.py`: builds all figures into `results/figures/` from `results.json`.
- `requirements.txt`: environment file.

## Reproduce
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python run_experiments.py 2>&1 | tee logs/run.log      # mkdir -p logs first
    python make_figures.py
Test the code on synthetic data first (takes ~2 min, numbers are meaningless):
    python run_experiments.py --smoke && python make_figures.py results_smoke

Full run is dominated by Q3 (about 10k tree fits on 240 x 4096). Expect from tens of minutes to a couple of hours
depending on cores. Bagging fits use all cores (results do not depend on core count).

## Design decisions (state these in the report)
- Split: per identity, `default_rng(26505).permutation` -> 6 train / 2 val / 2 test, saved once and reused.
- Tree selection: validation accuracy only; ties broken by fewer leaves. Bagging B selected on validation, ties -> smaller B.
- Bagging ensembles for different B are the first B trees of one 200-tree bag (nested).
- OOB accuracy uses only samples that have at least one OOB tree (coverage is stored in the results).
- Disagreement and vote margin are measured on the validation set. Vote ties break to the lowest class index.
- SAMME: a weak learner with weighted error >= 1 - 1/K is discarded and boosting stops (logged in `stop_reason`);
  a perfect learner (error 0) uses eps = 1e-12 and boosting stops. Toy 3-class, 10-sample round is in `results["q2a"]`.
- Boosting grid: one sequential 100-round run per depth, evaluated at 10/25/50/100 rounds (a prefix of a sequential run
  equals a separate run of that many rounds).
- Q3 depth study: shallow = 2, selected depth, deep = None. If the selected tree is already `max_depth=None`
  the script uses 2 / 16 / None; say so in the report, since nothing can be deeper than unrestricted.
- Q3 class vectors: tree = one-hot, bagging = vote fractions, boosting = normalised weighted vote.

"""Grouped cross-validation on the cached features: baselines, old-vs-new protocol, Phase 2 experiments,
Phase 3 error analysis. Every transform lives inside the sklearn Pipeline, so it is refit per training fold.

    python -m src.features   # once
    python -m src.evaluate   # -> results/results.csv, results/results.md, results/errors.md
"""
import logging
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.data import CLASSES, load_config, setup_logging
from src.features import load_cache, pool

log = logging.getLogger(__name__)


class RoundedRegressor(ClassifierMixin, BaseEstimator):
    """Ordinal approach: regress the class index (low=0, medium=1, high=2), round to the nearest class."""

    def __init__(self, reg=None):
        self.reg = reg

    def fit(self, X, y):
        self.reg_ = clone(self.reg).fit(X, y)
        self.classes_ = np.arange(len(CLASSES))
        return self

    def predict(self, X):
        return np.clip(np.rint(self.reg_.predict(X)), 0, len(CLASSES) - 1).astype(int)


def make_clf(name, seed=42):
    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=seed)
    return {
        "majority": lambda: DummyClassifier(strategy="most_frequent"),
        "stratified random": lambda: DummyClassifier(strategy="stratified", random_state=seed),
        "svm": lambda: make_pipeline(StandardScaler(), svm),
        "lda_svm": lambda: make_pipeline(StandardScaler(), LDA(n_components=2), svm),
        "lda": lambda: make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto")),
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced")),
        # C/gamma chosen by an inner grouped CV on the training folds only (nested CV)
        "svm_grid": lambda: GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 1e-4, 1e-3]},
            cv=StratifiedGroupKFold(3, shuffle=True, random_state=seed), scoring="f1_macro", n_jobs=-1),
        "ordinal_svr": lambda: RoundedRegressor(make_pipeline(StandardScaler(), SVR(C=10))),
    }[name]()


def scores(y, yp):
    return f1_score(y, yp, average="macro"), accuracy_score(y, yp), cohen_kappa_score(y, yp, weights="quadratic")


def run_cv(X, y, groups, clf, folds):
    oof, per_fold = np.empty_like(y), []
    for tr, te in folds:
        m = clone(clf)
        m.fit(X[tr], y[tr], groups=groups[tr]) if isinstance(m, GridSearchCV) else m.fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
        per_fold.append(scores(y[te], oof[te]))
    f = np.array(per_fold)
    return dict(macro_f1=f[:, 0].mean(), f1_std=f[:, 0].std(), accuracy=f[:, 1].mean(),
                qwk=f[:, 2].mean(), qwk_std=f[:, 2].std()), oof


def report(name, y, oof):
    log.info("--- %s ---\n%s\nconfusion matrix summed over folds (rows=true, cols=pred) %s\n%s", name,
             classification_report(y, oof, target_names=CLASSES, digits=3), CLASSES,
             confusion_matrix(y, oof, labels=range(len(CLASSES))))


def old_protocol(X, y, seed):
    """The original numbers: LDA fit on ALL clips, then one random 80/20 split (not grouped)."""
    Z = LDA(n_components=2).fit_transform(X, y)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    yp = SVC(kernel="rbf", C=10, class_weight="balanced", random_state=seed).fit(Z[tr], y[tr]).predict(Z[te])
    return scores(y[te], yp)


def random_split(X, y, seed):
    """Same single random split, but LDA inside the pipeline (fixes feature leakage, keeps speaker leakage)."""
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    return scores(y[te], make_clf("lda_svm", seed).fit(X[tr], y[tr]).predict(X[te]))


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
              for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def error_analysis(c, oof, name, out):
    y, groups, dur, paths = c["y"], c["groups"], c["duration"], c["paths"]
    ok = oof == y
    lines = [f"# Error analysis: {name} (out-of-fold predictions, grouped {len(set(groups))} recordings)\n",
             "## Confusion matrix (rows = true, cols = predicted)\n",
             to_markdown(pd.DataFrame(confusion_matrix(y, oof), index=CLASSES, columns=CLASSES).reset_index()
                         .rename(columns={"index": "true"}))]
    q = pd.qcut(dur, 4)
    by_dur = pd.DataFrame({"duration_s": q, "ok": ok, "y": y, "oof": oof}).groupby("duration_s", observed=True)
    lines += ["\n## By clip duration (quartiles)\n", to_markdown(by_dur.apply(lambda g: pd.Series({
        "clips": len(g), "accuracy": g.ok.mean(), "macro_f1": f1_score(g.y, g.oof, average="macro")}),
        include_groups=False).reset_index().astype({"duration_s": str, "clips": int}))]
    rec = pd.DataFrame({"recording": groups, "ok": ok}).groupby("recording").ok.agg(["size", "mean"])
    lines += ["\n## By recording\n",
              f"Per-recording accuracy: median {rec['mean'].median():.2f}, min {rec['mean'].min():.2f}, "
              f"max {rec['mean'].max():.2f}, std {rec['mean'].std():.2f}\n\nWorst 10 recordings:\n",
              to_markdown(rec.sort_values("mean").head(10).reset_index()
                          .rename(columns={"size": "clips", "mean": "accuracy"}))]
    # 10 errors to listen to: the low<->high confusions first, then random others
    rng = np.random.default_rng(0)
    wrong = np.flatnonzero(~ok)
    wrong = wrong[np.lexsort((rng.random(len(wrong)), -np.abs(oof[wrong] - y[wrong])))][:10]
    lines += ["\n## 10 misclassified clips to listen to\n", to_markdown(pd.DataFrame({
        "file": [os.path.basename(p) for p in paths[wrong]], "true": [CLASSES[i] for i in y[wrong]],
        "predicted": [CLASSES[i] for i in oof[wrong]], "duration_s": dur[wrong].astype(float)}))]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    log.info("wrote %s", out)


def main():
    setup_logging()
    cfg = load_config()
    seed = cfg["seed"]
    c = load_cache(cfg)
    y, groups, dur = c["y"], c["groups"], c["duration"]
    M, S = c["wavlm_mean"].astype(np.float32), c["wavlm_std"].astype(np.float32)
    last = M.shape[1] - 1

    log.info("dataset: %d clips, %d unique recordings", len(y), len(set(groups)))
    log.info("clips per class: %s", {k: int((y == i).sum()) for i, k in enumerate(CLASSES)})
    log.info("clips per recording: median %d, min %d, max %d",
             *(lambda v: (np.median(v), v.min(), v.max()))(pd.Series(groups).value_counts().to_numpy()))
    log.info("clip duration (s): %s", pd.Series(dur).describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_dict())
    log.info("clips longer than %ss (truncated): %d", cfg["max_seconds"], int((dur > cfg["max_seconds"]).sum()))

    folds = list(StratifiedGroupKFold(cfg["n_splits"], shuffle=True, random_state=seed).split(M, y, groups))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "a recording is in both train and test"

    rows, oofs = [], {}

    def add(phase, name, features, X, clf_name):
        r, oof = run_cv(X, y, groups, make_clf(clf_name, seed), folds)
        rows.append(dict(phase=phase, model=name, features=features, protocol="grouped 5-fold", **r))
        oofs[name] = (oof, features)
        log.info("%-40s macro F1 %.3f +/- %.3f  QWK %.3f", name, r["macro_f1"], r["f1_std"], r["qwk"])
        return r, oof

    # ---- Phase 1: before/after, baselines, grouped CV ----
    for name, fn in [("old: LDA on all data, 1 random split", old_protocol),
                     ("LDA in pipeline, 1 random split", random_split)]:
        f1, acc, qwk = fn(M[:, last], y, seed)
        rows.append(dict(phase=1, model=name, features=f"wavlm L{last} mean", protocol="single 80/20 split",
                         macro_f1=f1, f1_std=np.nan, accuracy=acc, qwk=qwk, qwk_std=np.nan))
        log.info("%-40s macro F1 %.3f (single split)", name, f1)
    for name, X, feat, clf in [("majority class", c["mfcc"], "-", "majority"),
                               ("stratified random", c["mfcc"], "-", "stratified random"),
                               ("MFCC + SVM", c["mfcc"], "40 MFCC mean+std", "svm"),
                               ("WavLM + Scaler/LDA/SVM", M[:, last], f"wavlm L{last} mean", "lda_svm")]:
        add(1, name, feat, X, clf)
        if clf in ("svm", "lda_svm"):
            report(name, y, oofs[name][0])

    # ---- Phase 2: layers, pooling, classifiers, ordinal ----
    layer_f1 = {L: add(2, f"layer {L} logreg", f"wavlm L{L} mean", M[:, L], "logreg")[0]["macro_f1"]
                for L in range(M.shape[1])}
    best_L = max(layer_f1, key=layer_f1.get)
    ms = add(2, f"layer {best_L} mean+std logreg", f"wavlm L{best_L} mean+std",
             pool(M, S, best_L, "mean+std"), "logreg")[0]["macro_f1"]
    best_pool = "mean+std" if ms > layer_f1[best_L] else "mean"
    X = pool(M, S, best_L, best_pool)
    feat = f"wavlm L{best_L} {best_pool}"
    for clf in ["lda_svm", "lda", "svm_grid", "ordinal_svr"]:
        add(2, f"layer {best_L} {best_pool} {clf}", feat, X, clf)
    if best_L != last:
        add(2, f"layer {last} mean svm_grid", f"wavlm L{last} mean", M[:, last], "svm_grid")

    res = pd.DataFrame(rows)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    res.to_csv(os.path.join(cfg["results_dir"], "results.csv"), index=False)
    with open(os.path.join(cfg["results_dir"], "results.md"), "w") as f:
        f.write(f"Grouped = StratifiedGroupKFold({cfg['n_splits']}) by recording id; mean over folds "
                f"(std = across folds). Layer and pooling were picked on these same folds, so the Phase 2 "
                f"best is slightly optimistic.\n\n" + to_markdown(res))
    log.info("\n%s", res.round(3).to_string(index=False))

    # ---- Phase 3: error analysis of the best grouped model ----
    best = res[res.protocol == "grouped 5-fold"].sort_values("macro_f1").iloc[-1].model
    report(best, y, oofs[best][0])
    error_analysis(c, oofs[best][0], f"{best} [{oofs[best][1]}]", os.path.join(cfg["results_dir"], "errors.md"))


if __name__ == "__main__":
    main()

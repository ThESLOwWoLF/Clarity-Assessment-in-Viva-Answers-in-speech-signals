"""Fit the final pipeline (layer / pooling / classifier from config.yaml) on ALL cached clips and save it.

    python -m src.train   # -> cfg['model_path']
"""
import logging
import os

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.data import CLASSES, load_config, setup_logging
from src.evaluate import make_clf
from src.features import load_cache, pool

log = logging.getLogger(__name__)


def main():
    setup_logging()
    cfg = load_config()
    c = load_cache(cfg)
    X = pool(c["wavlm_mean"].astype(np.float32), c["wavlm_std"].astype(np.float32), cfg["layer"], cfg["pooling"])
    clf = make_clf(cfg["classifier"], cfg["seed"])
    clf.set_params(**{k: True for k in clf.get_params() if k.endswith("svc__probability")})  # API returns probabilities
    clf.fit(X, c["y"], groups=c["groups"]) if isinstance(clf, GridSearchCV) else clf.fit(X, c["y"])
    if isinstance(clf, GridSearchCV):
        log.info("grid picked %s", clf.best_params_)
        clf = clf.best_estimator_
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    joblib.dump({"pipeline": clf, "classes": CLASSES, **{k: cfg[k] for k in
                 ("wavlm_model", "layer", "pooling", "max_seconds", "classifier")}}, cfg["model_path"])
    log.info("trained %s on %d clips, %s -> %s", cfg["classifier"], len(X), X.shape, cfg["model_path"])


if __name__ == "__main__":
    main()

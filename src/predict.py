"""wav file in -> label + class probabilities out. Loads WavLM and the classifier once.

    python -m src.predict "wav files/107_s1.wav" [--device cpu]
"""
import argparse
import json
import time

import joblib
import numpy as np

from src.data import load_config
from src.features import SR, Embedder, load_audio, pool


class Predictor:
    def __init__(self, model_path=None, device=None):
        self.art = joblib.load(model_path or load_config()["model_path"])
        self.embed = Embedder(self.art["wavlm_model"], device)
        self.clf = self.art["pipeline"]

    def predict_wav(self, wav):
        """wav: mono 16 kHz tensor. Same truncation as training."""
        mean, std = self.embed(wav[: int(self.art["max_seconds"] * SR)])
        x = pool(mean, std, self.art["layer"], self.art["pooling"])[None]
        classes = self.art["classes"]
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(x)[0]
            probs = {classes[i]: float(v) for i, v in zip(self.clf.classes_, p)}
            label = max(probs, key=probs.get)
        else:  # ordinal regressor: no probabilities
            label, probs = classes[int(self.clf.predict(x)[0])], None
        return label, probs

    def __call__(self, src):
        t0 = time.perf_counter()
        wav, sr = load_audio(src)
        label, probs = self.predict_wav(wav)
        return {"label": label, "probabilities": probs, "duration_s": round(len(wav) / SR, 3),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--device")
    a = ap.parse_args()
    p = Predictor(device=a.device)
    for w in a.wav:
        print(json.dumps({"file": w, **p(w)}))

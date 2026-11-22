"""API test with synthetic audio and a classifier fit on random vectors: no real data involved.
Downloads the WavLM checkpoint on first run."""
import io

import joblib
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.evaluate import make_clf


def wav_bytes(seconds=2.0, sr=44100, channels=2, fmt="WAV"):
    t = np.arange(int(seconds * sr)) / sr
    x = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    buf = io.BytesIO()
    sf.write(buf, np.repeat(x[:, None], channels, 1), sr, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    clf = make_clf("svm").set_params(svc__probability=True).fit(rng.standard_normal((60, 768)), np.arange(60) % 3)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"pipeline": clf, "classes": ["low", "medium", "high"], "wavlm_model": "microsoft/wavlm-base-plus",
                 "layer": 12, "pooling": "mean", "max_seconds": 8, "classifier": "svm"}, path)
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(path))
    mp.setenv("DEVICE", "cpu")
    from src.api import app
    with TestClient(app) as c:
        yield c
    mp.undo()


def test_predict_stereo_44k(client):
    r = client.post("/predict", files={"file": ("a.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["label"] in ("low", "medium", "high")
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-6
    assert abs(j["duration_s"] - 2.0) < 0.01 and j["latency_ms"] > 0


@pytest.mark.parametrize("data,code", [
    (b"not audio at all", 400),
    (wav_bytes(fmt="FLAC"), 400),
    (wav_bytes(sr=4000), 400),
    (wav_bytes(seconds=0.2), 400),
    (wav_bytes(seconds=61, sr=8000, channels=1), 413)])
def test_rejects_bad_input(client, data, code):
    r = client.post("/predict", files={"file": ("a.wav", data, "audio/wav")})
    assert r.status_code == code, r.text
    assert r.json()["detail"]

"""Run WavLM once over every clip and cache all-layer mean/std embeddings + MFCC stats.

    python -m src.features          # writes cfg['cache']; skips if cache already matches the clip list
"""
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMModel

from src.data import CLASSES, load_config, load_pairs, setup_logging

log = logging.getLogger(__name__)
SR = 16000


def load_audio(src):
    """Path or file-like -> (mono float32 tensor at 16 kHz, original sample rate)."""
    wav, sr = sf.read(src, dtype="float32", always_2d=True)  # raises on non-audio input
    wav = torch.from_numpy(wav.mean(1))  # stereo -> mono
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav, sr


class Embedder:
    """Frozen WavLM. One clip per forward pass: no padding, so embeddings don't depend on batch-mates
    (the CNN front end uses GroupNorm over time, which padding would change). An 8 s clip in fp16 uses <1 GB."""

    def __init__(self, model_name, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, wav):
        """wav: 1-D 16 kHz tensor -> (mean, std), each [n_layers+1, hidden] float32 numpy."""
        inp = self.fe(wav.numpy(), sampling_rate=SR, return_tensors="pt").to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            hs = torch.stack(self.model(**inp, output_hidden_states=True).hidden_states)[:, 0].float()  # [L, T, H]
        return hs.mean(1).cpu().numpy(), hs.std(1).cpu().numpy()


def mfcc_stats(wav, _mfcc=torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=40,
              melkwargs={"n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 64})):  # 25 ms / 10 ms frames, no empty mel bands
    m = _mfcc(wav)  # [40, frames]
    return torch.cat([m.mean(1), m.std(1)]).numpy()


def pool(mean, std, layer, pooling):
    """Pick one layer from cached [N, L, H] arrays (or [L, H] for one clip) and pool."""
    x = mean[..., layer, :]
    return np.concatenate([x, std[..., layer, :]], -1) if pooling == "mean+std" else x


def load_cache(cfg):
    z = np.load(cfg["cache"], allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    setup_logging()
    cfg = load_config()
    data = load_pairs(cfg)
    paths = data["path"].tolist()
    if os.path.exists(cfg["cache"]) and load_cache(cfg)["paths"].tolist() == paths:
        log.info("cache %s already matches %d clips, nothing to do", cfg["cache"], len(paths))
        return
    emb = Embedder(cfg["wavlm_model"])
    log.info("device %s, %d clips", emb.device, len(paths))
    max_len = int(cfg["max_seconds"] * SR)
    means, stds, mfccs, durs = [], [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        wav, _ = load_audio(p)
        durs.append(len(wav) / SR)  # full duration, before truncation
        wav = wav[:max_len]
        m, s = emb(wav)
        means.append(m.astype(np.float16))  # halves the cache; fp16 model output anyway
        stds.append(s.astype(np.float16))
        mfccs.append(mfcc_stats(wav))
        if i % 200 == 0:
            log.info("%d/%d  %.1f clips/s", i, len(paths), (i + 1) / (time.time() - t0))
    os.makedirs(os.path.dirname(cfg["cache"]), exist_ok=True)
    np.savez(
        cfg["cache"],
        paths=np.array(paths, dtype=object),
        y=data["label"].map(CLASSES.index).to_numpy(),
        groups=data["recording_id"].to_numpy(dtype=object),
        duration=np.array(durs, dtype=np.float32),
        wavlm_mean=np.stack(means),
        wavlm_std=np.stack(stds),
        mfcc=np.stack(mfccs),
    )
    log.info("saved %s (%.0f s)", cfg["cache"], time.time() - t0)


if __name__ == "__main__":
    main()

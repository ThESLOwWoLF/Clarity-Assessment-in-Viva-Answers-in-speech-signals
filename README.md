# Clarity Assessment in Viva Answers

Classifies viva (oral exam) answer clips as **low / medium / high** clarity using frozen
`microsoft/wavlm-base-plus` embeddings and a small sklearn classifier.

The dataset (audio + teacher labels) is private and is **not** in this repo. Point `config.yaml` at your local copy.

## Setup

```bash
python -m venv .venv
# GPU: pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Usage

```bash
python -m src.features     # WavLM once over all clips -> cache/features.npz (all 13 layers, mean+std, + MFCC). ~2 min on an RTX 4050
python -m src.evaluate     # grouped CV, baselines, experiments -> results/results.{csv,md}, results/errors.md
python -m src.train        # final model (layer/pooling/classifier from config.yaml) -> models/clarity.joblib
python -m src.predict "wav files/107_s1.wav"
uvicorn src.api:app --port 8000   # POST /predict (wav upload), demo UI at /demo
pytest                     # label parsing, recording id, API on synthetic audio
```

Docker (CPU): `docker build -t clarity .` then `docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" clarity`.
The model is mounted, not baked in: the SVM stores support vectors, which are embeddings of private clips.

## Results

3,563 labelled clips from 133 recordings (low 1659 / medium 1308 / high 596).
Evaluation: `StratifiedGroupKFold(5)` grouped by recording id, so no recording is in both train and test,
and every transform (scaler, LDA, ...) is fit on the training folds only. Mean ± std over folds.

| model | protocol | macro F1 | QWK |
|---|---|---|---|
| **old pipeline: LDA fit on all data, 1 random split** | single 80/20 | **0.691** | 0.611 |
| same, LDA inside the pipeline | single 80/20 | 0.424 | 0.232 |
| majority class | grouped 5-fold | 0.212 ± 0.002 | 0.000 |
| stratified random | grouped 5-fold | 0.331 ± 0.010 | 0.002 |
| clip duration only (logreg) | grouped 5-fold | 0.319 ± 0.012 | 0.115 |
| MFCC (40, mean+std) + SVM | grouped 5-fold | 0.415 ± 0.028 | 0.211 |
| WavLM L12 + Scaler/LDA/SVM (old model, honest eval) | grouped 5-fold | 0.425 ± 0.019 | 0.196 |
| WavLM L10 + logistic regression | grouped 5-fold | 0.453 ± 0.036 | 0.267 |
| **WavLM L10 + SVM, C/gamma by nested grouped CV (final)** | grouped 5-fold | **0.457 ± 0.032** | **0.327** |

Full table (all 13 layers, pooling, LDA alone, ordinal SVR): [results/results.md](results/results.md).
Layer and pooling were picked on the same folds, so the final number is slightly optimistic.

Latency per clip (clips are capped at 8 s, median 4.2 s, includes reading the file):
GPU RTX 4050 **39 ms** median (p90 63 ms), CPU **227 ms** median (p90 426 ms).

## Findings

1. **The reported 0.69 was leakage.** Fitting LDA (2 components, 3 classes) on all clips before splitting puts
   the test labels into the features. Moving LDA inside the pipeline drops the same split to 0.42. The honest
   grouped-CV score of the best model is **0.46 ± 0.03** macro F1.
2. **WavLM only slightly beats MFCC.** 0.457 vs 0.415, a gap of about one fold's standard deviation. Both are well above
   chance (0.33) and majority (0.21). Frozen WavLM features carry some clarity signal, but not much more than spectral stats.
3. **Medium is the hard class, and errors are often not adjacent.** Medium recall is 0.36, and 417 medium
   clips were called high. Ordinal regression did worse (0.37), so the labels don't behave like a clean scale
   in embedding space.
4. **Layer choice matters a little, pooling and LDA don't help.** All 13 layers score 0.42–0.45 with logreg (layer 10
   and layer 0 best). mean+std pooling and the LDA reduction both lower the score.
5. **Performance varies by recording, not by length.** Per-recording accuracy ranges from 0.12 to 0.92 (median 0.50).
   Short clips are mostly "low" (60%), but duration alone gives only chance-level F1 (0.32). Accuracy on short clips is higher
   only because the model predicts their majority class. Half of the clips are longer than 8 s and are truncated, so long answers are judged on their first 8 s.

Next steps worth trying: embed long clips in 8 s windows and average them instead of truncating; check
label agreement on a re-labelled subset (the ceiling may be label noise); LoRA fine-tuning only if those plateau.

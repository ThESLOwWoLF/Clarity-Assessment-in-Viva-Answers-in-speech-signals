Grouped = StratifiedGroupKFold(5) by recording id; mean over folds (std = across folds). Layer and pooling were picked on these same folds, so the Phase 2 best is slightly optimistic.

| phase | model | features | protocol | macro_f1 | f1_std | accuracy | qwk | qwk_std |
|---|---|---|---|---|---|---|---|---|
| 1 | old: LDA on all data, 1 random split | wavlm L12 mean | single 80/20 split | 0.691 | nan | 0.701 | 0.611 | nan |
| 1 | LDA in pipeline, 1 random split | wavlm L12 mean | single 80/20 split | 0.424 | nan | 0.454 | 0.232 | nan |
| 1 | majority class | - | grouped 5-fold | 0.212 | 0.002 | 0.466 | 0.000 | 0.000 |
| 1 | stratified random | - | grouped 5-fold | 0.331 | 0.010 | 0.385 | 0.002 | 0.020 |
| 1 | MFCC + SVM | 40 MFCC mean+std | grouped 5-fold | 0.415 | 0.028 | 0.489 | 0.211 | 0.066 |
| 1 | WavLM + Scaler/LDA/SVM | wavlm L12 mean | grouped 5-fold | 0.425 | 0.019 | 0.457 | 0.196 | 0.054 |
| 2 | layer 0 logreg | wavlm L0 mean | grouped 5-fold | 0.452 | 0.035 | 0.490 | 0.281 | 0.056 |
| 2 | layer 1 logreg | wavlm L1 mean | grouped 5-fold | 0.449 | 0.031 | 0.489 | 0.278 | 0.059 |
| 2 | layer 2 logreg | wavlm L2 mean | grouped 5-fold | 0.433 | 0.040 | 0.475 | 0.264 | 0.063 |
| 2 | layer 3 logreg | wavlm L3 mean | grouped 5-fold | 0.445 | 0.022 | 0.485 | 0.264 | 0.052 |
| 2 | layer 4 logreg | wavlm L4 mean | grouped 5-fold | 0.435 | 0.021 | 0.478 | 0.261 | 0.040 |
| 2 | layer 5 logreg | wavlm L5 mean | grouped 5-fold | 0.436 | 0.014 | 0.476 | 0.247 | 0.038 |
| 2 | layer 6 logreg | wavlm L6 mean | grouped 5-fold | 0.443 | 0.031 | 0.482 | 0.252 | 0.058 |
| 2 | layer 7 logreg | wavlm L7 mean | grouped 5-fold | 0.442 | 0.018 | 0.482 | 0.250 | 0.040 |
| 2 | layer 8 logreg | wavlm L8 mean | grouped 5-fold | 0.430 | 0.019 | 0.472 | 0.257 | 0.035 |
| 2 | layer 9 logreg | wavlm L9 mean | grouped 5-fold | 0.424 | 0.026 | 0.465 | 0.259 | 0.053 |
| 2 | layer 10 logreg | wavlm L10 mean | grouped 5-fold | 0.453 | 0.036 | 0.492 | 0.267 | 0.064 |
| 2 | layer 11 logreg | wavlm L11 mean | grouped 5-fold | 0.448 | 0.027 | 0.487 | 0.272 | 0.070 |
| 2 | layer 12 logreg | wavlm L12 mean | grouped 5-fold | 0.440 | 0.028 | 0.478 | 0.256 | 0.062 |
| 2 | layer 10 mean+std logreg | wavlm L10 mean+std | grouped 5-fold | 0.425 | 0.023 | 0.468 | 0.209 | 0.047 |
| 2 | layer 10 mean lda_svm | wavlm L10 mean | grouped 5-fold | 0.406 | 0.020 | 0.438 | 0.165 | 0.019 |
| 2 | layer 10 mean lda | wavlm L10 mean | grouped 5-fold | 0.431 | 0.033 | 0.499 | 0.222 | 0.069 |
| 2 | layer 10 mean svm_grid | wavlm L10 mean | grouped 5-fold | 0.457 | 0.032 | 0.506 | 0.327 | 0.077 |
| 2 | layer 10 mean ordinal_svr | wavlm L10 mean | grouped 5-fold | 0.373 | 0.023 | 0.461 | 0.214 | 0.052 |
| 2 | layer 12 mean svm_grid | wavlm L12 mean | grouped 5-fold | 0.451 | 0.030 | 0.500 | 0.308 | 0.063 |

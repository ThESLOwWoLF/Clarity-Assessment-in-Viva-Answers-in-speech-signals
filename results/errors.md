# Error analysis: layer 10 mean svm_grid [wavlm L10 mean] (out-of-fold predictions, grouped 133 recordings)

## Confusion matrix (rows = true, cols = predicted)

| true | low | medium | high |
|---|---|---|---|
| low | 1101 | 351 | 207 |
| medium | 422 | 469 | 417 |
| high | 169 | 197 | 230 |


## By clip duration (quartiles)

| duration_s | clips | accuracy | macro_f1 |
|---|---|---|---|
| (0.283, 3.147] | 891 | 0.616 | 0.419 |
| (3.147, 7.49] | 891 | 0.493 | 0.455 |
| (7.49, 17.264] | 890 | 0.476 | 0.456 |
| (17.264, 319.48] | 891 | 0.435 | 0.431 |


## By recording

Per-recording accuracy: median 0.50, min 0.12, max 0.92, std 0.16

Worst 10 recordings:

| recording | clips | accuracy |
|---|---|---|
| 107 | 8 | 0.125 |
| 183 | 16 | 0.125 |
| 223 | 45 | 0.178 |
| 246 | 20 | 0.200 |
| 236 | 24 | 0.208 |
| 228 | 28 | 0.214 |
| 225 | 41 | 0.268 |
| 242 | 43 | 0.279 |
| 240 | 25 | 0.280 |
| 253 | 25 | 0.280 |


## 10 misclassified clips to listen to

| file | true | predicted | duration_s |
|---|---|---|---|
| 101_s32.wav | low | high | 18.728 |
| 245_s8.wav | high | low | 12.726 |
| 246_s12.wav | high | low | 61.703 |
| 256_s25.wav | low | high | 13.455 |
| 144_s32.wav | low | high | 3.562 |
| 269_s43.wav | low | high | 5.548 |
| 153_s10.wav | high | low | 0.812 |
| 256_s18.wav | low | high | 5.479 |
| 242_s13.wav | low | high | 29.486 |
| 153_s41.wav | high | low | 1.943 |

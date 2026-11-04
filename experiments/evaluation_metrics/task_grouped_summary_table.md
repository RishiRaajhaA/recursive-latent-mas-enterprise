# Cross-Round Task Performance & Computational Summary Table

Grouped by Task IDs across recursion depths ($r=0, 1, 2, 3$) for both **Text ReAct** and **Latent ReAct**, including Latency, FLOPs, Accuracy, Output Tokens, and Aggregate Rows.

### Task 24 (Easy Tier)
| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 24 | Easy | `latent_react` | $r = 0$ | 4.38s | 4.80 TFLOPs | 40.0% | 68 tok | **FAIL** |
| 24 | Easy | `latent_react` | $r = 1$ | 8.50s | 12.59 TFLOPs | 40.0% | 197 tok | **FAIL** |
| 24 | Easy | `latent_react` | $r = 2$ | 13.05s | 22.84 TFLOPs | 40.0% | 283 tok | **FAIL** |
| 24 | Easy | `latent_react` | $r = 3$ | 19.19s | 33.22 TFLOPs | 40.0% | 385 tok | **FAIL** |
| **Task 24** | **Easy** | **`latent_react` Sub-Total** | **$r = 0..3$ Avg** | **11.28s** | **18.36 TFLOPs** | **40.0%** | **233 tok** | **0.0% Win Rate** |
| 24 | Easy | `text_react` | $r = 0$ | 6.98s | 5.30 TFLOPs | 40.0% | 94 tok | **FAIL** |
| 24 | Easy | `text_react` | $r = 1$ | 11.33s | 14.76 TFLOPs | 40.0% | 240 tok | **FAIL** |
| 24 | Easy | `text_react` | $r = 2$ | 17.03s | 28.35 TFLOPs | 40.0% | 338 tok | **FAIL** |
| 24 | Easy | `text_react` | $r = 3$ | 22.50s | 41.07 TFLOPs | 40.0% | 480 tok | **FAIL** |
| **Task 24** | **Easy** | **`text_react` Sub-Total** | **$r = 0..3$ Avg** | **14.46s** | **22.37 TFLOPs** | **40.0%** | **288 tok** | **0.0% Win Rate** |

### Task 57 (Easy Tier)
| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 57 | Easy | `latent_react` | $r = 0$ | 4.83s | 10.56 TFLOPs | 100.0% | 93 tok | **SUCCESS** |
| 57 | Easy | `latent_react` | $r = 1$ | 9.36s | 26.17 TFLOPs | 100.0% | 186 tok | **SUCCESS** |
| 57 | Easy | `latent_react` | $r = 2$ | 12.35s | 41.78 TFLOPs | 100.0% | 279 tok | **SUCCESS** |
| 57 | Easy | `latent_react` | $r = 3$ | 18.41s | 57.39 TFLOPs | 100.0% | 372 tok | **SUCCESS** |
| **Task 57** | **Easy** | **`latent_react` Sub-Total** | **$r = 0..3$ Avg** | **11.24s** | **33.98 TFLOPs** | **100.0%** | **232 tok** | **100.0% Win Rate** |
| 57 | Easy | `text_react` | $r = 0$ | 5.95s | 11.25 TFLOPs | 100.0% | 125 tok | **SUCCESS** |
| 57 | Easy | `text_react` | $r = 1$ | 14.14s | 33.51 TFLOPs | 100.0% | 268 tok | **SUCCESS** |
| 57 | Easy | `text_react` | $r = 2$ | 19.85s | 55.31 TFLOPs | 100.0% | 385 tok | **SUCCESS** |
| 57 | Easy | `text_react` | $r = 3$ | 24.09s | 77.11 TFLOPs | 100.0% | 502 tok | **SUCCESS** |
| **Task 57** | **Easy** | **`text_react` Sub-Total** | **$r = 0..3$ Avg** | **16.01s** | **44.29 TFLOPs** | **100.0%** | **320 tok** | **100.0% Win Rate** |

### Task 100 (Medium Tier)
| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 | Medium | `latent_react` | $r = 0$ | 4.77s | 5.92 TFLOPs | 60.0% | 116 tok | **FAIL** |
| 100 | Medium | `latent_react` | $r = 1$ | 9.68s | 14.67 TFLOPs | 60.0% | 225 tok | **FAIL** |
| 100 | Medium | `latent_react` | $r = 2$ | 16.20s | 25.98 TFLOPs | 60.0% | 334 tok | **FAIL** |
| 100 | Medium | `latent_react` | $r = 3$ | 21.22s | 37.29 TFLOPs | 60.0% | 443 tok | **FAIL** |
| **Task 100** | **Medium** | **`latent_react` Sub-Total** | **$r = 0..3$ Avg** | **12.97s** | **20.96 TFLOPs** | **60.0%** | **280 tok** | **0.0% Win Rate** |
| 100 | Medium | `text_react` | $r = 0$ | 6.71s | 6.80 TFLOPs | 60.0% | 164 tok | **FAIL** |
| 100 | Medium | `text_react` | $r = 1$ | 13.80s | 17.21 TFLOPs | 60.0% | 306 tok | **FAIL** |
| 100 | Medium | `text_react` | $r = 2$ | 23.02s | 34.88 TFLOPs | 60.0% | 502 tok | **FAIL** |
| 100 | Medium | `text_react` | $r = 3$ | 31.78s | 50.42 TFLOPs | 60.0% | 660 tok | **FAIL** |
| **Task 100** | **Medium** | **`text_react` Sub-Total** | **$r = 0..3$ Avg** | **18.83s** | **27.33 TFLOPs** | **60.0%** | **408 tok** | **0.0% Win Rate** |

### Task 67 (Medium Tier)
| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 67 | Medium | `latent_react` | $r = 0$ | 3.53s | 4.52 TFLOPs | 24.0% | 84 tok | **FAIL** |
| 67 | Medium | `latent_react` | $r = 1$ | 9.49s | 11.88 TFLOPs | 32.0% | 226 tok | **FAIL** |
| 67 | Medium | `latent_react` | $r = 2$ | 15.03s | 22.66 TFLOPs | 100.0% | 334 tok | **SUCCESS** |
| 67 | Medium | `latent_react` | $r = 3$ | 20.62s | 35.16 TFLOPs | 100.0% | 442 tok | **SUCCESS** |
| **Task 67** | **Medium** | **`latent_react` Sub-Total** | **$r = 0..3$ Avg** | **12.17s** | **18.56 TFLOPs** | **64.0%** | **272 tok** | **50.0% Win Rate** |
| 67 | Medium | `text_react` | $r = 0$ | 5.23s | 5.40 TFLOPs | 24.0% | 116 tok | **FAIL** |
| 67 | Medium | `text_react` | $r = 1$ | 12.96s | 14.85 TFLOPs | 32.0% | 296 tok | **FAIL** |
| 67 | Medium | `text_react` | $r = 2$ | 18.58s | 30.88 TFLOPs | 40.0% | 416 tok | **FAIL** |
| 67 | Medium | `text_react` | $r = 3$ | 28.62s | 54.61 TFLOPs | 40.0% | 589 tok | **FAIL** |
| **Task 67** | **Medium** | **`text_react` Sub-Total** | **$r = 0..3$ Avg** | **16.35s** | **26.44 TFLOPs** | **34.0%** | **354 tok** | **0.0% Win Rate** |

### Task 7 (Hard Tier)
| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 7 | Hard | `latent_react` | $r = 0$ | 3.31s | 5.12 TFLOPs | 67.0% | 75 tok | **FAIL** |
| 7 | Hard | `latent_react` | $r = 1$ | 6.44s | 10.45 TFLOPs | 67.0% | 143 tok | **FAIL** |
| 7 | Hard | `latent_react` | $r = 2$ | 10.55s | 15.79 TFLOPs | 67.0% | 211 tok | **FAIL** |
| 7 | Hard | `latent_react` | $r = 3$ | 12.45s | 21.12 TFLOPs | 67.0% | 279 tok | **FAIL** |
| **Task 7** | **Hard** | **`latent_react` Sub-Total** | **$r = 0..3$ Avg** | **8.19s** | **13.12 TFLOPs** | **67.0%** | **177 tok** | **0.0% Win Rate** |
| 7 | Hard | `text_react` | $r = 0$ | 5.29s | 5.95 TFLOPs | 67.0% | 118 tok | **FAIL** |
| 7 | Hard | `text_react` | $r = 1$ | 9.42s | 12.07 TFLOPs | 67.0% | 196 tok | **FAIL** |
| 7 | Hard | `text_react` | $r = 2$ | 12.15s | 18.20 TFLOPs | 67.0% | 274 tok | **FAIL** |
| 7 | Hard | `text_react` | $r = 3$ | 17.13s | 24.32 TFLOPs | 67.0% | 352 tok | **FAIL** |
| **Task 7** | **Hard** | **`text_react` Sub-Total** | **$r = 0..3$ Avg** | **11.00s** | **15.14 TFLOPs** | **67.0%** | **235 tok** | **0.0% Win Rate** |

### Task 8 (Hard Tier)
| Task ID | Tier | Architecture | Recursion Depth ($r$) | Latency (s) | Compute (TFLOPs) | Accuracy (%) | Output Tokens | Outcome |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | Hard | `latent_react` | $r = 0$ | 3.34s | 5.05 TFLOPs | 67.0% | 75 tok | **FAIL** |
| 8 | Hard | `latent_react` | $r = 1$ | 8.62s | 10.89 TFLOPs | 67.0% | 208 tok | **FAIL** |
| 8 | Hard | `latent_react` | $r = 2$ | 17.94s | 26.33 TFLOPs | 80.0% | 380 tok | **SUCCESS** |
| 8 | Hard | `latent_react` | $r = 3$ | 24.41s | 52.10 TFLOPs | 93.0% | 505 tok | **SUCCESS** |
| **Task 8** | **Hard** | **`latent_react` Sub-Total** | **$r = 0..3$ Avg** | **13.58s** | **23.59 TFLOPs** | **76.8%** | **292 tok** | **50.0% Win Rate** |
| 8 | Hard | `text_react` | $r = 0$ | 6.49s | 6.14 TFLOPs | 67.0% | 157 tok | **FAIL** |
| 8 | Hard | `text_react` | $r = 1$ | 14.94s | 16.32 TFLOPs | 73.0% | 349 tok | **SUCCESS** |
| 8 | Hard | `text_react` | $r = 2$ | 18.95s | 28.34 TFLOPs | 73.0% | 448 tok | **SUCCESS** |
| 8 | Hard | `text_react` | $r = 3$ | 24.13s | 40.13 TFLOPs | 73.0% | 521 tok | **SUCCESS** |
| **Task 8** | **Hard** | **`text_react` Sub-Total** | **$r = 0..3$ Avg** | **16.13s** | **22.73 TFLOPs** | **71.5%** | **369 tok** | **75.0% Win Rate** |

## Overall Matrix Summary & Architecture Aggregates

| Architecture | Recursion Depth ($r$) | Mean Latency (s) | Mean Compute (TFLOPs) | Mean Accuracy (%) | Mean Output Tokens | Success Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `latent_react` | $r = 0$ | 4.03s | 6.00 TFLOPs | 59.7% | 85 tok | **16.7%** |
| `latent_react` | $r = 1$ | 8.68s | 14.44 TFLOPs | 61.0% | 198 tok | **16.7%** |
| `latent_react` | $r = 2$ | 14.19s | 25.90 TFLOPs | 74.5% | 304 tok | **50.0%** |
| `latent_react` | $r = 3$ | 19.38s | 39.38 TFLOPs | 76.7% | 404 tok | **50.0%** |
| `text_react` | $r = 0$ | 6.11s | 6.81 TFLOPs | 59.7% | 129 tok | **16.7%** |
| `text_react` | $r = 1$ | 12.77s | 18.12 TFLOPs | 62.0% | 276 tok | **33.3%** |
| `text_react` | $r = 2$ | 18.26s | 32.66 TFLOPs | 63.3% | 394 tok | **33.3%** |
| `text_react` | $r = 3$ | 24.71s | 47.94 TFLOPs | 63.3% | 517 tok | **33.3%** |
| **`latent_react` Total** | **All Rounds ($r=0..3$)** | **11.57s** | **21.43 TFLOPs** | **68.0%** | **248 tok** | **33.3%** |
| **`text_react` Total** | **All Rounds ($r=0..3$)** | **15.46s** | **26.38 TFLOPs** | **62.1%** | **329 tok** | **29.2%** |
| **Grand Total (48 Runs)** | **Overall Matrix** | **13.52s** | **23.91 TFLOPs** | **65.0%** | **288 tok** | **31.2%** |
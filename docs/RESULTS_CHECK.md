# Verified Results

The archived outputs were checked against the revised manuscript.

## Publication gate

- Synthetic runs expected/completed: 90/90
- Controlled family-imbalance check: passed
- Family-composition rows: 1350
- Threshold-sensitivity rows: 630
- Family-prior rows: 450
- Lorenz-96 rows: 100
- CausalTime rows: 684

## Family-composition mean F1

| Composition | Family-normalized | Equal-method | Simple majority |
|---|---:|---:|---:|
| 1:1:1 | 0.746032 | 0.746032 | 0.746032 |
| 2:1:1 | 0.746032 | 0.709259 | 0.709259 |
| 3:1:1 | 0.733439 | 0.689630 | 0.762222 |

## Threshold-free external evaluation

Pooled over 77 evaluable external targets:

- Average Precision: 0.485601 vs 0.463497
- Mean AP difference: +0.022104
- Paired Wilcoxon p = 0.000039

- ROC-AUC: 0.637292 vs 0.627533
- Mean ROC-AUC difference: +0.009759
- Paired Wilcoxon p = 0.033351

## Holm-corrected domain result

CausalTime-PM2.5 Average Precision:

- Family-normalized: 0.550629
- Equal-method: 0.522842
- Raw p = 0.000184
- Holm-adjusted p = 0.001470

The method is therefore interpreted primarily as a continuous reliability/ranking representation rather than as a universally superior fixed-threshold binary selector.

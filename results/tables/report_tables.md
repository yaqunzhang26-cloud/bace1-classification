# Report tables

## Table 1. Cross-validation results

| Fingerprint | Model | CV_Accuracy | CV_Sensitivity | CV_Specificity | CV_F1 | CV_ROC_AUC |
|---|---|---|---|---|---|---|
| Morgan | Random Forest | 0.8542 ± 0.0106 | 0.8873 ± 0.0106 | 0.7779 ± 0.0193 | 0.8946 ± 0.0078 | 0.9149 ± 0.0075 |
| MACCS | Random Forest | 0.8430 ± 0.0078 | 0.8840 ± 0.0095 | 0.7486 ± 0.0190 | 0.8871 ± 0.0058 | 0.8987 ± 0.0068 |
| MACCS | Logistic Regression | 0.7779 ± 0.0198 | 0.7886 ± 0.0186 | 0.7532 ± 0.0266 | 0.8320 ± 0.0157 | 0.8453 ± 0.0130 |
| MACCS | Linear SVM | 0.7790 ± 0.0190 | 0.7910 ± 0.0159 | 0.7511 ± 0.0290 | 0.8331 ± 0.0147 | 0.8451 ± 0.0128 |
| Morgan | Logistic Regression | 0.8019 ± 0.0165 | 0.8535 ± 0.0187 | 0.6828 ± 0.0250 | 0.8573 ± 0.0125 | 0.8435 ± 0.0091 |
| Morgan | Linear SVM | 0.7812 ± 0.0173 | 0.8300 ± 0.0185 | 0.6687 ± 0.0168 | 0.8410 ± 0.0134 | 0.8176 ± 0.0115 |

## Table 2. Held-out test metrics

| Fingerprint | Model | CV_ROC_AUC | Accuracy | Sensitivity | Specificity | F1 | ROC_AUC | TN | FP | FN | TP | Test_n | Train_n | Threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Morgan | Random Forest | 0.9149 | 0.8581 | 0.8816 | 0.8040 | 0.8965 | 0.9199 | 398 | 97 | 135 | 1005 | 1635 | 6537 | 6.0 |

## Table 3. Confusion matrix

|  | Predicted Inactive | Predicted Active |
|---|---|---|
| True Inactive | 398 | 97 |
| True Active | 135 | 1005 |

## Table 4. Predicted Active candidates

| compound_id | predicted_active_score | smiles |
|---|---|---|
| cmpd2558 | 0.5492 | COC1=C(NC(=O)C2CCN(C(=O)C3=CC4=C(C=C3)N(C)N=N4)CC2)C=CC=C1... |
| cmpd3636 | 0.5260 | CC1=NC=C(C(=O)N[C@@H]2CCC[C@@H]2C(=O)NC2=CC(F)=C(C)C=C2)N=C1... |
| cmpd1098 | 0.5200 | COC1CN(C2=CC(F)=CC=C2)C12CCN(CC1=CC=C(C#N)C=C1)CC2... |
| cmpd2586 | 0.5144 | CC(C)CCN1C(=O)C2=C(N=CN2CC2=CC(C#N)=CC=C2)N(C2=C(F)C=C(F)C=C2)C1=O... |

## Table 5. Threshold sensitivity

| threshold | n_active | active_fraction |
|---|---|---|
| 0.3 | 1062 | 26.5566% |
| 0.4 | 119 | 2.9757% |
| 0.5 | 4 | 0.1000% |
| 0.6 | 0 | 0.0000% |
| 0.7 | 0 | 0.0000% |
| 0.8 | 0 | 0.0000% |

## Table 6. Nearest-neighbor Tanimoto summary

| n_new | mean | median | q25 | q75 | min | max | fraction_lt_0.3 | fraction_lt_0.5 |
|---|---|---|---|---|---|---|---|---|
| 3999 | 0.2932 | 0.2857 | 0.2581 | 0.3191 | 0.1519 | 0.8478 | 61.6904% | 99.4249% |
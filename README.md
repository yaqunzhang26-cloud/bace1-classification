# Machine Learning Classification and Virtual Screening of BACE1 Inhibitors

End-to-end machine learning pipeline for BACE1 inhibitor classification and virtual screening (AIDD8122 final project).

## Overview

- **Dataset**: 8,172 labeled BACE1 compounds (5,700 active / 2,472 inactive, pIC50 >= 6.0) from ChEMBL, plus 3,999 unlabeled screening compounds.
- **Featurization**: MACCS keys (166 bits) and Morgan fingerprints (radius = 2, 2,048 bits).
- **Benchmark**: 2 fingerprints × 3 algorithms = six fingerprint–algorithm combinations, stratified 5-fold CV.
- **Final model**: Random Forest + Morgan fingerprints (CV ROC-AUC 0.9149; test ROC-AUC 0.9199 on n = 1,635).
- **Application**: applicability domain analysis (nearest-neighbor Tanimoto + PCA) and virtual screening of 3,999 compounds, nominating four candidate inhibitors at threshold 0.5.

## Repository Structure

- `data/raw/` — labeled & screening CSV files
- `data/processed/` — CV metrics, test metrics, predictions
- `src/` — complete pipeline script
- `results/figures/` — Figures 1–11

## Setup & Reproduction


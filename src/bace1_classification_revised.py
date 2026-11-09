"""BACE1 inhibitor classification coursework.

Inputs (place beside this script):
  data/raw/BACE1_compounds_clean.csv
  data/raw/new_compounds.csv

Outputs:
  data/processed/cv_results.csv                  5-fold CV metrics for every model/fingerprint pair
  data/processed/test_set_metrics.csv            one held-out test evaluation of the CV-selected pipeline
  data/processed/new_compounds_predictions.csv   predictions for all valid new compounds

Important assumption: pIC50 >= 6.0 is labelled Active (1), and pIC50 < 6.0
Inactive (0). The assignment does not specify a cutoff; confirm with the instructor.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("default")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer, recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

TRAIN_FILE = "data/raw/BACE1_compounds_clean.csv"
NEW_FILE = "data/raw/new_compounds.csv"
THRESHOLD = 6.0
RANDOM_STATE = 42


def fingerprint(smiles, kind):
    """Return a NumPy bit vector, or None if the SMILES cannot be parsed."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if kind == "Morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    elif kind == "MACCS":
        fp = MACCSkeys.GenMACCSKeys(mol)
    else:
        raise ValueError(f"Unknown fingerprint type: {kind}")
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def fingerprint_matrix(smiles, kind):
    rows, valid_indices = [], []
    for i, smi in enumerate(smiles):
        arr = fingerprint(smi, kind)
        if arr is not None:
            rows.append(arr)
            valid_indices.append(i)
    if not rows:
        return np.empty((0, 0), dtype=np.uint8), np.asarray([], dtype=int)
    return np.asarray(rows), np.asarray(valid_indices, dtype=int)


def calculate_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_score),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def build_models():
    # Scaling is fitted inside each CV fold through Pipeline, avoiding leakage.
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        "Linear SVM": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def get_scores(model, X):
    """Return continuous scores for ROC-AUC for either probabilistic or margin models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def main():
    for filename in (TRAIN_FILE, NEW_FILE):
        if not Path(filename).is_file():
            raise FileNotFoundError(
                f"Cannot find {filename}. Put it in the same folder as this script."
            )

    data = pd.read_csv(TRAIN_FILE)
    new_data = pd.read_csv(NEW_FILE)
    required_train = {"smiles", "pIC50"}
    required_new = {"smiles", "compound_id"}
    if not required_train.issubset(data.columns):
        raise ValueError(f"Training CSV must contain columns: {sorted(required_train)}")
    if not required_new.issubset(new_data.columns):
        raise ValueError(f"New-compounds CSV must contain columns: {sorted(required_new)}")

    data["pIC50"] = pd.to_numeric(data["pIC50"], errors="coerce")
    before = len(data)
    data = data.dropna(subset=["smiles", "pIC50"]).copy().reset_index(drop=True)
    data["activity"] = (data["pIC50"] >= THRESHOLD).astype(int)
    if data["activity"].nunique() != 2:
        raise ValueError("The chosen pIC50 threshold did not produce both classes.")
    y = data["activity"].to_numpy()

    # Fixed independent hold-out set. All model/fingerprint selection uses only train split CV.
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_by_fp = {}
    for kind in ("Morgan", "MACCS"):
        X, valid = fingerprint_matrix(data["smiles"], kind)
        if len(valid) != len(data):
            raise ValueError(f"Invalid training SMILES for {kind}: {len(data) - len(valid)}")
        X_by_fp[kind] = X

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    specificity_scorer = make_scorer(recall_score, pos_label=0, zero_division=0)
    scoring = {
        "accuracy": "accuracy",
        "sensitivity": "recall",
        "specificity": specificity_scorer,
        "f1": "f1",
        "auc": "roc_auc",
    }

    # Compare all 3 classifiers x 2 fingerprints using the same 5-fold CV protocol.
    cv_rows = []
    models = build_models()
    for fp_name, X in X_by_fp.items():
        for model_name, model in models.items():
            result = cross_validate(
                model, X[train_idx], y[train_idx], cv=cv, scoring=scoring,
                n_jobs=1, error_score="raise", return_train_score=False,
            )
            row = {"Fingerprint": fp_name, "Model": model_name}
            for key in scoring:
                vals = result[f"test_{key}"]
                row[f"CV_{key}_mean"] = float(np.mean(vals))
                row[f"CV_{key}_std"] = float(np.std(vals, ddof=1))
            cv_rows.append(row)
    cv_table = pd.DataFrame(cv_rows).sort_values(
        ["CV_auc_mean", "CV_f1_mean"], ascending=False
    ).reset_index(drop=True)
    cv_table.to_csv("data/processed/cv_results.csv", index=False)

    # Select solely on training-set CV ROC-AUC, then evaluate once on the held-out test set.
    best = cv_table.iloc[0]
    best_fp, best_model_name = best["Fingerprint"], best["Model"]
    selected_model = clone(models[best_model_name])
    X_selected = X_by_fp[best_fp]
    selected_model.fit(X_selected[train_idx], y[train_idx])
    y_test_pred = selected_model.predict(X_selected[test_idx])
    y_test_score = get_scores(selected_model, X_selected[test_idx])
    test_metrics = calculate_metrics(y[test_idx], y_test_pred, y_test_score)
    test_row = {
        "Selected_Fingerprint": best_fp,
        "Selected_Model": best_model_name,
        "Selection_CV_ROC_AUC_mean": float(best["CV_auc_mean"]),
        **test_metrics,
        "Test_n": int(len(test_idx)),
        "Train_n": int(len(train_idx)),
        "Activity_threshold_pIC50": THRESHOLD,
    }
    pd.DataFrame([test_row]).to_csv("data/processed/test_set_metrics.csv", index=False)

    # Refit selected model on all labelled training data for final new-compound screening.
    final_model = clone(models[best_model_name])
    final_model.fit(X_selected, y)
    X_new, valid_new = fingerprint_matrix(new_data["smiles"], best_fp)
    if len(valid_new) != len(new_data):
        bad = len(new_data) - len(valid_new)
        raise ValueError(
            f"Found {bad} invalid SMILES in new dataset. No partial prediction file was written."
        )
    pred_class = final_model.predict(X_new)
    pred_score = get_scores(final_model, X_new)
    out = new_data.copy()
    out["predicted_activity_class"] = np.where(pred_class == 1, "Active", "Inactive")
    out["predicted_active_score"] = pred_score
    out["selected_model"] = best_model_name
    out["selected_fingerprint"] = best_fp
    out["activity_threshold_pIC50"] = THRESHOLD
    out.to_csv("data/processed/new_compounds_predictions.csv", index=False)

    print(f"Rows loaded: {before}; rows retained after missing-value removal: {len(data)}")
    print(f"Activity threshold assumption: pIC50 >= {THRESHOLD} -> Active")
    print("Class counts:", data["activity"].value_counts().to_dict())
    print("\nCross-validation results (all model/fingerprint pairs):")
    print(cv_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSelected model and held-out test metrics:")
    print(pd.DataFrame([test_row]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nNew-compound prediction counts:")
    print(out["predicted_activity_class"].value_counts().to_string())
    print("\nOutputs written: data/processed/cv_results.csv, data/processed/test_set_metrics.csv, data/processed/new_compounds_predictions.csv")
    print("Note: random train/test splitting can overestimate generalisation when related/duplicate structures occur across splits.")


if __name__ == "__main__":
    main()

"""Generate Figure 1 workflow and Table 1 using the revised BACE1 outputs.

Expected files beside this script:
  data/raw/BACE1_compounds_clean.csv
  new_compounds.csv
  data/processed/new_compounds_predictions.csv

Outputs:
  figures/fig1_workflow_revised.png
  tables/table1_dataset_summary_revised.csv
  tables/table1_dataset_summary_revised.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split

from bace1_classification_revised import THRESHOLD, RANDOM_STATE

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "results/figures"
TABLE_DIR = BASE_DIR / "results/tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

TRAIN_FILE = BASE_DIR / "data/raw/BACE1_compounds_clean.csv"
NEW_FILE = BASE_DIR / "data/raw/new_compounds.csv"
# Explicitly use the revised outputs; do not silently fall back to legacy files.
PREDICTION_FILE = BASE_DIR / "data/processed/new_compounds_predictions.csv"
DPI = 300
plt.rcParams["font.family"] = "DejaVu Sans"


def df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def load_and_validate():
    for path in (TRAIN_FILE, NEW_FILE, PREDICTION_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required input file not found: {path}")

    data = pd.read_csv(TRAIN_FILE)
    new_data = pd.read_csv(NEW_FILE)
    pred = pd.read_csv(PREDICTION_FILE)

    if not {"smiles", "pIC50"}.issubset(data.columns):
        raise ValueError("Training data must contain 'smiles' and 'pIC50' columns.")
    if not {"compound_id", "smiles"}.issubset(new_data.columns):
        raise ValueError("New-compounds data must contain 'compound_id' and 'smiles'.")
    required_pred = {
        "compound_id", "smiles", "predicted_activity_class", "selected_model",
        "selected_fingerprint", "activity_threshold_pIC50",
    }
    if not required_pred.issubset(pred.columns):
        raise ValueError(
            "The selected prediction file does not have the expected revised-output columns: "
            + ", ".join(sorted(required_pred - set(pred.columns)))
        )

    data["pIC50"] = pd.to_numeric(data["pIC50"], errors="coerce")
    # Match the main modelling script's cleaning rule exactly.
    data = data.dropna(subset=["smiles", "pIC50"]).copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("No labelled training records remain after the main script's cleaning rule.")
    y = (data["pIC50"] >= THRESHOLD).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError("The configured activity threshold did not produce both classes.")

    if len(new_data) != len(pred):
        raise ValueError(f"New-compound rows ({len(new_data)}) do not match prediction rows ({len(pred)}).")
    # Check both IDs and SMILES in row order to prevent mixing output versions or reordered files.
    if not new_data["compound_id"].astype(str).reset_index(drop=True).equals(
        pred["compound_id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("Prediction compound_id order does not match data/processed/new_compounds.csv.")
    if not new_data["smiles"].astype(str).reset_index(drop=True).equals(
        pred["smiles"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("Prediction SMILES order does not match data/processed/new_compounds.csv.")

    threshold_values = pd.to_numeric(pred["activity_threshold_pIC50"], errors="coerce").dropna().unique()
    if len(threshold_values) != 1 or not np.isclose(threshold_values[0], THRESHOLD):
        raise ValueError("Prediction threshold does not match THRESHOLD in the main modelling script.")
    model_names = pred["selected_model"].dropna().astype(str).unique()
    fp_names = pred["selected_fingerprint"].dropna().astype(str).unique()
    if len(model_names) != 1 or len(fp_names) != 1:
        raise ValueError("Prediction output contains inconsistent or missing model/fingerprint metadata.")

    return data, y, new_data, pred, model_names[0], fp_names[0]


def make_table1():
    data, y, new_data, pred, model_name, fp_name = load_and_validate()
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20,
        random_state=RANDOM_STATE, stratify=y,
    )
    y_train, y_test = y[train_idx], y[test_idx]
    pred_active = pred["predicted_activity_class"].astype(str).str.strip().str.lower().eq("active")
    if not pred["predicted_activity_class"].astype(str).str.strip().str.lower().isin({"active", "inactive"}).all():
        raise ValueError("Prediction file has unexpected activity class labels.")
    n_active_new = int(pred_active.sum())

    rows = [
        {"Dataset": "Training set (80%; observed labels)", "N": len(train_idx),
         "Active": int(y_train.sum()), "Inactive": int((y_train == 0).sum()),
         "Active fraction": f"{y_train.mean():.2%}", "Label source": "Observed pIC50"},
        {"Dataset": "Held-out test set (20%; observed labels)", "N": len(test_idx),
         "Active": int(y_test.sum()), "Inactive": int((y_test == 0).sum()),
         "Active fraction": f"{y_test.mean():.2%}", "Label source": "Observed pIC50"},
        {"Dataset": "New compounds (model-predicted)", "N": len(new_data),
         "Active": n_active_new, "Inactive": len(new_data) - n_active_new,
         "Active fraction": f"{n_active_new / len(new_data):.2%}",
         "Label source": f"Predicted by {model_name} ({fp_name})"},
        {"Dataset": "Total labelled data", "N": len(data),
         "Active": int(y.sum()), "Inactive": int((y == 0).sum()),
         "Active fraction": f"{y.mean():.2%}", "Label source": "Observed pIC50"},
    ]
    df = pd.DataFrame(rows)
    csv_path = TABLE_DIR / "table1_dataset_summary_revised.csv"
    md_path = TABLE_DIR / "table1_dataset_summary_revised.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df_to_md(df), encoding="utf-8")
    print("\nTable 1: dataset summary (revised output files)")
    print(df.to_string(index=False))
    print(f"\nThreshold: pIC50 >= {THRESHOLD}; random_state={RANDOM_STATE}; split=80/20 stratified")
    print(f"Selected prediction metadata: {model_name} + {fp_name}")
    print(f"Saved: {csv_path}\nSaved: {md_path}")


def make_figure1():
    # Layout makes the independent test evaluation and new-compound screening parallel branches.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fontsize=9, face="#E8F1F5"):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.08,rounding_size=0.12",
                               linewidth=1.25, edgecolor="#2F4F4F", facecolor=face)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
        return (x, y, w, h)

    def arrow(start, end, rad=0.0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     connectionstyle=f"arc3,rad={rad}",
                                     mutation_scale=13, linewidth=1.25, color="#2F4F4F"))

    box(4.2, 8.85, 3.6, 0.65, "Labelled BACE1 dataset\nBACE1_compounds_clean.csv", 9.2)
    box(4.2, 7.65, 3.6, 0.72, f"Clean missing labels and assign classes\npIC50 ≥ {THRESHOLD}: Active; otherwise Inactive", 8.8)
    box(4.2, 6.4, 3.6, 0.72, "Stratified train/test split\n80% training / 20% held-out test", 9)
    box(0.45, 4.75, 3.5, 0.85, "Training split only\nMorgan (2048 bits) and MACCS (167 bits)", 8.8)
    box(0.45, 3.35, 3.5, 0.85, "5-fold stratified cross-validation\n3 classifiers × 2 fingerprints", 8.8)
    box(4.25, 3.35, 3.5, 0.85, "Select best combination\nby mean CV ROC-AUC", 9)
    box(4.25, 1.65, 3.5, 0.85, "Evaluate once on held-out test set\n(final test metrics)", 8.8, face="#F8F0DF")
    box(8.05, 4.75, 3.5, 0.85, "New compounds\ndata/processed/new_compounds.csv", 9)
    box(8.05, 3.15, 3.5, 0.95, "Refit selected model on all labelled data\nthen predict new compounds", 8.5, face="#EAF3E8")

    # Main preprocessing and split.
    arrow((6.0, 8.85), (6.0, 8.37))
    arrow((6.0, 7.65), (6.0, 7.12))
    # Only training data enter CV and model selection.
    arrow((4.2, 6.65), (3.95, 5.2))
    arrow((2.2, 4.75), (2.2, 4.2))
    arrow((3.95, 3.78), (4.25, 3.78))
    # Test evaluation receives the held-out test data and the CV-selected model.
    arrow((7.8, 6.65), (7.2, 2.5), rad=-0.12)
    arrow((6.0, 3.35), (6.0, 2.5))
    # New-compound prediction is a separate branch; it does not flow from test evaluation.
    arrow((7.8, 3.75), (8.05, 3.62), rad=0.0)
    arrow((9.8, 4.75), (9.8, 4.1))

    ax.set_title("Figure 1. Overall workflow of BACE1 inhibitor classification", fontsize=12, pad=14)
    fig.tight_layout()
    out = FIG_DIR / "fig1_workflow_revised.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_table1()
    make_figure1()

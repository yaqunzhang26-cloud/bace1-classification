"""
Generate report-ready figures and tables for the BACE1 classification coursework.

Inputs (same folder):
  data/processed/cv_results.csv
  data/processed/test_set_metrics.csv
  new_compounds_predictions.csv
  data/raw/BACE1_compounds_clean.csv
  data/raw/new_compounds.csv
  bace1_classification_revised.py   (for fingerprint_matrix / build_models)

Outputs:
  figures/*.png
  tables/*.csv
  tables/*.md
  tables/report_tables.md
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from bace1_classification_revised import (
    fingerprint_matrix, build_models, THRESHOLD, RANDOM_STATE
)

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
DPI = 300


# ---------- helpers ----------
def df_to_md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def morgan_matrix(smiles, n_bits=2048):
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows)


def nearest_neighbor_tanimoto(X_train, X_new, block=256):
    """Max Tanimoto similarity of each new compound to the training set."""
    X_train_f = X_train.astype(np.float32)
    train_sum = X_train_f.sum(axis=1)
    out = []
    for start in range(0, X_new.shape[0], block):
        Xb = X_new[start:start + block].astype(np.float32)
        inter = Xb @ X_train_f.T
        union = Xb.sum(axis=1)[:, None] + train_sum[None, :] - inter
        sim = inter / np.clip(union, 1e-9, None)
        out.append(sim.max(axis=1))
    return np.concatenate(out)


# ---------- figure functions ----------
def fig_cv_auc(cv: pd.DataFrame):
    df = cv.copy()
    df["label"] = df["Fingerprint"] + " + " + df["Model"]
    df = df.sort_values("CV_auc_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["label"], df["CV_auc_mean"], color="#4C72B0")
    for b, v in zip(bars, df["CV_auc_mean"]):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)
    ax.set_xlim(0.75, max(0.95, df["CV_auc_mean"].max() + 0.02))
    ax.set_xlabel("5-fold CV ROC-AUC (mean)")
    ax.set_title("Cross-validation ROC-AUC by fingerprint/model")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_cv_auc_bar.png", dpi=DPI)
    plt.close(fig)


def fig_cv_heatmap(cv: pd.DataFrame):
    df = cv.copy()
    df["label"] = df["Fingerprint"] + " + " + df["Model"]
    df = df.sort_values("CV_auc_mean", ascending=False)
    metrics = ["CV_accuracy_mean", "CV_sensitivity_mean",
               "CV_specificity_mean", "CV_f1_mean", "CV_auc_mean"]
    names = ["Accuracy", "Sensitivity", "Specificity", "F1", "ROC-AUC"]
    mat = df[metrics].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.65, vmax=0.95)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.88 else "black", fontsize=8)
    ax.set_title("Cross-validation metrics (mean)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_cv_metrics_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig_test_confusion(test: pd.Series):
    tn, fp, fn, tp = int(test["TN"]), int(test["FP"]), int(test["FN"]), int(test["TP"])
    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    labels = ["Inactive", "Active"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (Accuracy={test['Accuracy']:.4f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_test_confusion_matrix.png", dpi=DPI)
    plt.close(fig)


def fig_test_metrics(test: pd.Series):
    names = ["Accuracy", "Sensitivity", "Specificity", "F1", "ROC_AUC"]
    vals = [float(test[n]) for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Held-out test set performance (Morgan + Random Forest)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_test_metrics_bar.png", dpi=DPI)
    plt.close(fig)


def fig_new_score_hist(pred: pd.DataFrame):
    scores = pred["predicted_active_score"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, color="#4C72B0", edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", label="Default threshold = 0.5")
    ax.set_xlabel("Predicted active score")
    ax.set_ylabel("Number of new compounds")
    ax.set_title(f"Prediction score distribution (n={len(scores)})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_new_score_hist.png", dpi=DPI)
    plt.close(fig)


def fig_threshold_sensitivity(pred: pd.DataFrame):
    scores = pred["predicted_active_score"].astype(float).to_numpy()
    thresholds = np.arange(0.05, 0.96, 0.01)
    counts = np.array([(scores >= t).sum() for t in thresholds])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, counts, color="#C44E52", linewidth=2)
    ax.axvline(0.5, color="gray", linestyle="--", label="Threshold = 0.5")
    i05 = int(np.argmin(np.abs(thresholds - 0.5)))
    ax.scatter([0.5], [counts[i05]], color="red", zorder=5)
    ax.annotate(f"{counts[i05]} Active at 0.5", (0.5, counts[i05]),
                textcoords="offset points", xytext=(10, 10), fontsize=9)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Number of compounds predicted Active")
    ax.set_title("Threshold sensitivity on new compounds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_threshold_sensitivity.png", dpi=DPI)
    plt.close(fig)


def fig_nn_similarity(sim: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sim, bins=50, color="#55A868", edgecolor="white")
    ax.axvline(np.mean(sim), color="red", linestyle="--",
               label=f"Mean = {np.mean(sim):.3f}")
    ax.axvline(np.median(sim), color="orange", linestyle=":",
               label=f"Median = {np.median(sim):.3f}")
    ax.set_xlabel("Max Tanimoto similarity to labelled compounds (Morgan)")
    ax.set_ylabel("Number of new compounds")
    ax.set_title("Chemical-space proximity of new compounds")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_nearest_neighbor_tanimoto.png", dpi=DPI)
    plt.close(fig)


def fig_pca_chemical_space(X_train, X_new, y_train):
    X_all = np.vstack([X_train, X_new]).astype(np.float32)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_all)
    Z_train, Z_new = Z[:len(X_train)], Z[len(X_train):]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Z_train[y_train == 1, 0], Z_train[y_train == 1, 1],
               s=6, alpha=0.45, c="#4C72B0", label="Labelled Active")
    ax.scatter(Z_train[y_train == 0, 0], Z_train[y_train == 0, 1],
               s=6, alpha=0.45, c="#DD8452", label="Labelled Inactive")
    ax.scatter(Z_new[:, 0], Z_new[:, 1], s=6, alpha=0.35,
               c="black", label="New compounds")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of Morgan fingerprints: labelled vs new compounds")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_pca_chemical_space.png", dpi=DPI)
    plt.close(fig)
    return pca.explained_variance_ratio_


def fig_test_roc(X_train, y_train, train_idx, test_idx):
    model = build_models()["Random Forest"]
    model.fit(X_train[train_idx], y_train[train_idx])
    y_score = model.predict_proba(X_train[test_idx])[:, 1]
    y_true = y_train[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test-set ROC curve (Morgan + Random Forest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_test_roc.png", dpi=DPI)
    plt.close(fig)
    pd.DataFrame({"y_true": y_true, "y_score": y_score}).to_csv(
        TABLE_DIR / "test_predictions.csv", index=False)
    return roc_auc


# ---------- main ----------
def main():
    cv = pd.read_csv("data/processed/cv_results.csv")
    test = pd.read_csv("data/processed/test_set_metrics.csv").iloc[0]
    pred = pd.read_csv("data/processed/new_compounds_predictions.csv")

    # ---- report tables ----
    cv_report = cv.copy()
    cv_report["CV_Accuracy"] = cv_report.apply(
        lambda r: f"{r['CV_accuracy_mean']:.4f} ± {r['CV_accuracy_std']:.4f}", axis=1)
    cv_report["CV_Sensitivity"] = cv_report.apply(
        lambda r: f"{r['CV_sensitivity_mean']:.4f} ± {r['CV_sensitivity_std']:.4f}", axis=1)
    cv_report["CV_Specificity"] = cv_report.apply(
        lambda r: f"{r['CV_specificity_mean']:.4f} ± {r['CV_specificity_std']:.4f}", axis=1)
    cv_report["CV_F1"] = cv_report.apply(
        lambda r: f"{r['CV_f1_mean']:.4f} ± {r['CV_f1_std']:.4f}", axis=1)
    cv_report["CV_ROC_AUC"] = cv_report.apply(
        lambda r: f"{r['CV_auc_mean']:.4f} ± {r['CV_auc_std']:.4f}", axis=1)
    cv_report = cv_report[["Fingerprint", "Model", "CV_Accuracy", "CV_Sensitivity",
                           "CV_Specificity", "CV_F1", "CV_ROC_AUC"]]
    cv_report.to_csv(TABLE_DIR / "table1_cv_results.csv", index=False)
    (TABLE_DIR / "table1_cv_results.md").write_text(df_to_md(cv_report), encoding="utf-8")

    test_report = pd.DataFrame([{
        "Fingerprint": test["Selected_Fingerprint"],
        "Model": test["Selected_Model"],
        "CV_ROC_AUC": f"{test['Selection_CV_ROC_AUC_mean']:.4f}",
        "Accuracy": f"{test['Accuracy']:.4f}",
        "Sensitivity": f"{test['Sensitivity']:.4f}",
        "Specificity": f"{test['Specificity']:.4f}",
        "F1": f"{test['F1']:.4f}",
        "ROC_AUC": f"{test['ROC_AUC']:.4f}",
        "TN": int(test["TN"]), "FP": int(test["FP"]),
        "FN": int(test["FN"]), "TP": int(test["TP"]),
        "Test_n": int(test["Test_n"]), "Train_n": int(test["Train_n"]),
        "Threshold": test["Activity_threshold_pIC50"],
    }])
    test_report.to_csv(TABLE_DIR / "table2_test_metrics.csv", index=False)
    (TABLE_DIR / "table2_test_metrics.md").write_text(df_to_md(test_report), encoding="utf-8")

    cm = pd.DataFrame([
        {"": "True Inactive", "Predicted Inactive": int(test["TN"]), "Predicted Active": int(test["FP"])},
        {"": "True Active", "Predicted Inactive": int(test["FN"]), "Predicted Active": int(test["TP"])},
    ])
    cm.to_csv(TABLE_DIR / "table3_confusion_matrix.csv", index=False)
    (TABLE_DIR / "table3_confusion_matrix.md").write_text(df_to_md(cm), encoding="utf-8")

    active = pred[pred["predicted_activity_class"].str.lower() == "active"].copy()
    active = active.sort_values("predicted_active_score", ascending=False)
    active_report = active[["compound_id", "predicted_active_score", "smiles"]].copy()
    active_report["predicted_active_score"] = active_report["predicted_active_score"].map(lambda x: f"{x:.4f}")
    active_report["smiles"] = active_report["smiles"].astype(str).str.slice(0, 80) + "..."
    active_report.to_csv(TABLE_DIR / "table4_active_candidates.csv", index=False)
    (TABLE_DIR / "table4_active_candidates.md").write_text(df_to_md(active_report), encoding="utf-8")

    scores = pred["predicted_active_score"].astype(float).to_numpy()
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n = int((scores >= t).sum())
        rows.append({"threshold": t, "n_active": n,
                     "active_fraction": f"{n / len(scores):.4%}"})
    thr_df = pd.DataFrame(rows)
    thr_df.to_csv(TABLE_DIR / "table5_threshold_sensitivity.csv", index=False)
    (TABLE_DIR / "table5_threshold_sensitivity.md").write_text(df_to_md(thr_df), encoding="utf-8")

    # ---- load original data for similarity / PCA / ROC ----
    data = pd.read_csv("data/raw/BACE1_compounds_clean.csv")
    data["pIC50"] = pd.to_numeric(data["pIC50"], errors="coerce")
    data = data.dropna(subset=["smiles", "pIC50"]).reset_index(drop=True)
    y = (data["pIC50"] >= THRESHOLD).astype(int).to_numpy()

    X_train, valid_train = fingerprint_matrix(data["smiles"], "Morgan")
    if len(valid_train) != len(data):
        raise ValueError("Invalid SMILES in training data.")

    new_data = pd.read_csv("data/raw/new_compounds.csv")
    X_new, valid_new = fingerprint_matrix(new_data["smiles"], "Morgan")
    if len(valid_new) != len(new_data):
        raise ValueError("Invalid SMILES in new compounds.")

    sim = nearest_neighbor_tanimoto(X_train, X_new)
    nn_summary = pd.DataFrame([{
        "n_new": len(sim),
        "mean": f"{np.mean(sim):.4f}",
        "median": f"{np.median(sim):.4f}",
        "q25": f"{np.quantile(sim, 0.25):.4f}",
        "q75": f"{np.quantile(sim, 0.75):.4f}",
        "min": f"{np.min(sim):.4f}",
        "max": f"{np.max(sim):.4f}",
        "fraction_lt_0.3": f"{(sim < 0.3).mean():.4%}",
        "fraction_lt_0.5": f"{(sim < 0.5).mean():.4%}",
    }])
    nn_summary.to_csv(TABLE_DIR / "table6_nearest_neighbor_summary.csv", index=False)
    (TABLE_DIR / "table6_nearest_neighbor_summary.md").write_text(
        df_to_md(nn_summary), encoding="utf-8")

    # ---- figures ----
    fig_cv_auc(cv)
    fig_cv_heatmap(cv)
    fig_test_confusion(test)
    fig_test_metrics(test)
    fig_new_score_hist(pred)
    fig_threshold_sensitivity(pred)
    fig_nn_similarity(sim)
    pca_var = fig_pca_chemical_space(X_train, X_new, y)

    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    roc_auc = fig_test_roc(X_train, y, train_idx, test_idx)

    # ---- combined markdown ----
    combined = "\n\n".join([
        "# Report tables",
        "## Table 1. Cross-validation results",
        df_to_md(cv_report),
        "## Table 2. Held-out test metrics",
        df_to_md(test_report),
        "## Table 3. Confusion matrix",
        df_to_md(cm),
        "## Table 4. Predicted Active candidates",
        df_to_md(active_report) if len(active_report) else "_No Active predictions._",
        "## Table 5. Threshold sensitivity",
        df_to_md(thr_df),
        "## Table 6. Nearest-neighbor Tanimoto summary",
        df_to_md(nn_summary),
    ])
    (TABLE_DIR / "report_tables.md").write_text(combined, encoding="utf-8")

    print("Figures written to:", FIG_DIR.resolve())
    print("Tables written to:", TABLE_DIR.resolve())
    print(f"Re-trained test ROC-AUC: {roc_auc:.4f}")
    print(f"PCA explained variance: PC1={pca_var[0]:.3f}, PC2={pca_var[1]:.3f}")
    print(f"New compound score quantiles: "
          f"q50={np.median(scores):.4f}, q90={np.quantile(scores, 0.90):.4f}, "
          f"max={scores.max():.4f}")
    print(f"NN Tanimoto: mean={sim.mean():.4f}, median={np.median(sim):.4f}, "
          f"<0.3={(sim < 0.3).mean():.2%}, <0.5={(sim < 0.5).mean():.2%}")


if __name__ == "__main__":
    main()
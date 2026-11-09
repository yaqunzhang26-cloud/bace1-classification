"""候选分子结构图 + 适用域检查
运行: cd ~/CourseWork/BACE1-AIDD8122 && python src/bace1_candidate_check.py
"""
from pathlib import Path
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)  # 与主管线一致

def locate(name):
    """在当前目录和上级目录递归找文件（跳过 venv）"""
    for base in (Path.cwd(), Path.cwd().parent):
        for p in base.rglob(name):
            if "venv" not in p.parts:
                return p
    raise FileNotFoundError(f"没找到 {name}，把 ls 输出发给我")

def to_fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    return None if m is None else mfpgen.GetFingerprint(m)

# 1) 训练集指纹库（自动定位）
train_path = locate("BACE1_compounds_clean.csv")
train = pd.read_csv(train_path)
smi_col = next(c for c in train.columns if c.lower() == "smiles")
train_fps = [fp for fp in (to_fp(s) for s in train[smi_col]) if fp]
print(f"训练集: {train_path} | 有效指纹 {len(train_fps)}")

# 2) 候选 = 预测分 >= 0.5（完整 SMILES，避开 md 截断）
pred_path = locate("new_compounds_predictions.csv")
pred = pd.read_csv(pred_path)
score_col = next(c for c in pred.columns if "score" in c.lower())
id_col = next(c for c in pred.columns if "compound" in c.lower())
cand = pred[pred[score_col] >= 0.5].copy()
print(f"候选文件: {pred_path} | 候选数 {len(cand)}")
assert len(cand) == 4, "候选数不是 4，停下来检查！"

# 3) 2x2 结构图
fig_path = Path("results/figures/fig10_candidate_structures.png")
fig_path.parent.mkdir(parents=True, exist_ok=True)
mols, legends = [], []
for _, r in cand.iterrows():
    mols.append(Chem.MolFromSmiles(r[smi_col]))
    legends.append(f"{r[id_col]}\nscore={r[score_col]:.3f}")
Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(400, 400), legends=legends).save(fig_path)
print(f"结构图 -> {fig_path}")

# 4) 最近邻 Tanimoto（适用域）
tab_path = Path("results/tables/table7_candidate_applicability.csv")
tab_path.parent.mkdir(parents=True, exist_ok=True)
rows = []
for _, r in cand.iterrows():
    sims = DataStructs.BulkTanimotoSimilarity(to_fp(r[smi_col]), train_fps)
    rows.append({"compound_id": r[id_col], "score": r[score_col],
                 "nn_tanimoto": max(sims), "mean_tanimoto": sum(sims) / len(sims)})
out = pd.DataFrame(rows).sort_values("nn_tanimoto", ascending=False)
out.to_csv(tab_path, index=False)
print(out.to_string(index=False))
print(f"适用域表 -> {tab_path}")



import os, json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from multimodal_vit import MultiModalViT_timm


gold_csv     = r""
manifest_csv = r""
outputs_dir  = r""

ckpt_path    = r""
local_vit    = r""
tab_stats    = r""

BATCH_SIZE   = 16
AGG_MODE     = "max"
THR_FOR_VOTE = 0.5
TAB_COLS     = ["age","sex","esr","crp","b27"]


img_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def equalize_rgb(img: Image.Image) -> Image.Image:

    if img.mode != "RGB":
        img = img.convert("RGB")
    r, g, b = img.split()
    r = ImageOps.equalize(r)
    g = ImageOps.equalize(g)
    b = ImageOps.equalize(b)
    return Image.merge("RGB", (r, g, b))

def load_tab_stats(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def norm_tabular(arr, stats):

    if stats is None:
        return arr

    if "mean" in stats and isinstance(stats["mean"], dict):
        means = np.array([stats["mean"].get(c, 0.0) for c in TAB_COLS], dtype=np.float32)
        stds  = np.array([stats["std"].get(c, 1.0)  for c in TAB_COLS], dtype=np.float32)
    else:

        means = np.array(stats.get("mean", [0]*len(TAB_COLS)), dtype=np.float32)
        stds  = np.array(stats.get("std",  [1]*len(TAB_COLS)), dtype=np.float32)
    stds = np.where(stds == 0, 1.0, stds)
    return (arr - means) / stds

def read_csv_auto(path):
    for enc in ["utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)


class ExtMMDataset(Dataset):
    def __init__(self, df, stats=None):
        self.df = df.reset_index(drop=True)
        self.stats = stats

    def __len__(self): return len(self.df)

    def _make_tab(self, row):

        sex = str(row["sex"]).strip().upper()
        sex = 1.0 if sex in ["M","MALE","男"] else 0.0
        vals = np.array([
            float(row["age"]),
            sex,
            float(row["esr"]),
            float(row["crp"]),
            float(row["b27"]),
        ], dtype=np.float32)
        vals = norm_tabular(vals, self.stats)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        img = Image.open(row["image_path"]).convert("RGB")
        img = equalize_rgb(img)
        img = img_transform(img)
        # 2) 表型
        tab = self._make_tab(row)
        return {
            "image": img,
            "tab": tab,
            "patient_id": row["patient_id"]
        }


def aggregate_patient(df_img, mode="max", thr=0.5):
    g = df_img.groupby("patient_id")["prob_AS"]
    if mode == "max":
        s = g.max()
    elif mode == "mean":
        s = g.mean()
    elif mode == "vote":
        s = g.apply(lambda x: (x >= thr).astype(int).mean())
    else:
        raise ValueError("unknown mode")
    return s.reset_index().rename(columns={"prob_AS":"prob_AS"})


def main():
    os.makedirs(outputs_dir, exist_ok=True)


    gold = read_csv_auto(gold_csv).rename(columns={
        "ID":"patient_id",
        "诊断1=as，0=非AS":"label",
        "AGE":"age", "SEX":"sex",
        "ESR":"esr", "CRP":"crp", "B27":"b27"
    })
    keep = ["patient_id","label"] + TAB_COLS
    gold = gold[keep].copy()


    for c in ["age","esr","crp","b27"]:
        gold[c] = pd.to_numeric(gold[c], errors="coerce").fillna(0.0)
    gold["sex"] = gold["sex"].astype(str).str.strip()


    mani = read_csv_auto(manifest_csv)
    df = pd.merge(mani, gold, on="patient_id", how="inner")
    if len(df) == 0:
        raise RuntimeError("manifest 与 gold 无交集，请检查 patient_id 对齐。")
    print(f"样本量（交集）: {len(df)}")


    stats = load_tab_stats(tab_stats)


    ds = ExtMMDataset(df, stats=stats)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalViT_timm(
        num_tabular_features=len(TAB_COLS),
        local_vit_weight_path=local_vit
    ).to(device)


    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("✅ 已严格加载权重。eval() 模式已启用。")

    probs, pids = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            tab = batch["tab"].to(device)
            logits = model(img, tab)
            pr = torch.softmax(logits, dim=1)[:,1]
            probs.append(pr.cpu().numpy())
            pids += list(batch["patient_id"])
    probs = np.concatenate(probs)


    df_img = pd.DataFrame({"patient_id": pids, "prob_AS": probs})
    df_img = df_img.merge(gold[["patient_id","label"]], on="patient_id", how="left")
    df_img.to_csv(os.path.join(outputs_dir, "predictions_image_level.csv"),
                  index=False, encoding="utf-8-sig")

    df_pat = aggregate_patient(df_img, mode=AGG_MODE, thr=THR_FOR_VOTE)
    df_pat = df_pat.merge(gold[["patient_id","label"]], on="patient_id", how="left")
    df_pat.to_csv(os.path.join(outputs_dir, "predictions_patient_level.csv"),
                  index=False, encoding="utf-8-sig")

    print("概率前20:", probs[:20])
    print("均值/方差:", float(np.mean(probs)), float(np.var(probs)))
    print("✅ 推理完成。")
    print("保存到：", os.path.join(outputs_dir, "predictions_patient_level.csv"))

if __name__ == "__main__":
    main()

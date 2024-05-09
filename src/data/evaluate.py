# -*- coding: utf-8 -*
import glob
import os

import pandas as pd


def update_evaluations(dirpath: str) -> pd.DataFrame:
    ret = []
    for path in glob.glob(os.path.join(dirpath, "*.csv.gz")):
        df = pd.read_csv(path, compression="gzip")
        ret.append({
            "path": path,
            "phash": os.path.basename(path).split("_")[-2],
            "model": os.path.basename(path).split("_")[-3],
        } | do_evaluation(df))
    ret = pd.DataFrame(ret).sort_values("accuracy_presence")
    ret.to_csv(os.path.join(dirpath, "results.csv"), index=False)
    return ret


def do_evaluation(df: pd.DataFrame) -> dict[str, float]:
    def fn(row: pd.Series) -> pd.Series:
        pred_num_hedges = 0
        pred_hedges = []
        for line in row.generation.split("\n"):
            if line.startswith("Number of Hedges: ") and line.split()[-1].isdigit():
                pred_num_hedges = int(line.split()[-1])
            if line.startswith("List of Hedges: "):
                ln = line.replace("List of Hedges: ", "")[1:-1]
                pred_hedges = [h.replace('"', "") for h in ln.split(", ")]
        row["pred_num_hedges"] = pred_num_hedges
        row["pred_hedges"] = pred_hedges
        return row
    df = df.assign(hedges=df.hedges.fillna(""))
    df = df.apply(fn, axis=1)
    df = df.assign(
        hedge_present=df.num_hedges > 0,
        pred_hedge_present=df.pred_num_hedges > 0
    )

    def add_jaccard(row: pd.Series) -> pd.Series:
        preds = set([h.lower() for h in row.pred_hedges])
        refs = set([h.lower() for h in row.hedges])
        if len(preds | refs) == 0:
            row["jaccard"] = 1.0
        else:
            row["jaccard"] = len(preds & refs) / len(preds | refs)
        return row
    # NOTE: Some hedge annotations have parentheses around them.
    df = df.assign(hedges=df.hedges.str.replace("(", "").str.replace(")", ""))
    df = df.assign(hedges=df.hedges.str.split(", ").fillna("").apply(list))
    df = df.apply(add_jaccard, axis=1)

    accuracy_presence = sum(df.hedge_present == df.pred_hedge_present) / len(df)
    accuracy_count = sum(df.num_hedges == df.pred_num_hedges) / len(df)
    jaccard = df.jaccard.mean()
    return {
        "accuracy_presence": accuracy_presence,
        "accuracy_count": accuracy_count,
        "jaccard": jaccard,
        "count": len(df),
    }

# -*- coding: utf-8 -*
import glob
import os

import pandas as pd


def update_evaluations(dirpath: str) -> None:
    ret = []
    for path in glob.glob(os.path.join(dirpath, "*.csv.gz")):
        df = pd.read_csv(path, compression="gzip")
        ret.append({
            "path": path,
        } | do_evaluation(df))
    pd.DataFrame(ret).sort_values("accuracy_presence").to_csv(
        os.path.join(dirpath, "results.csv"), index=False
    )


def do_evaluation(df: pd.DataFrame) -> dict[str, float]:
    def fn(row: pd.Series) -> pd.Series:
        pred_num_hedges = 0
        pred_hedges = []
        for line in row.generation.split("\n"):
            if line.startswith("Number of Hedges: "):
                pred_num_hedges = int(line.split()[-1])
            if line.startswith("List of Hedges: "):
                ln = line.replace("List of Hedges: ", "")[1:-1]
                pred_hedges = [h.replace('"', "") for h in ln.split(", ")]
        row["pred_num_hedges"] = pred_num_hedges
        row["pred_hedges"] = pred_hedges
        return row
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
    df = df.assign(hedges=df.hedges.str.split(", ").fillna("").apply(list))
    df = df.apply(add_jaccard, axis=1)

    accuracy_presence = sum(df.hedge_present == df.pred_hedge_present) / len(df)
    accuracy_count = sum(df.num_hedges == df.pred_num_hedges) / len(df)
    jaccard = df.jaccard.mean()
    return {
        "accuracy_presence": accuracy_presence,
        "accuracy_count": accuracy_count,
        "jaccard": jaccard
    }

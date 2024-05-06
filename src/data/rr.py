# -*- coding: utf-8 -*
import os

import pandas as pd

from ..core.path import dirparent


RR_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "rr")


def load() -> pd.DataFrame:
    ret = pd.read_csv(
        os.path.join(RR_DIR, "RoadrunnerOneFile.csv")
    ).rename(columns={"Unnamed: 0": "Index", "Transcript": "Utterance"})
    ret.columns = [c.lower() for c in ret.columns]
    ret = ret.assign(
        num_hedges=ret.hedges.str.split(", ").str.len().fillna(0).astype(int)
    )
    return ret

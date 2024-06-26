#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run OpenAI models zero-shot on our generated questions.

Usage Examples:
    $ openai_zero_shot.py                   # No args needed.
    $ openai_zero_shot.py -o path/to/outdir # Custom outdir.
    $ openai_zero_shot.py --model gpt-4 --temperature 0.75
"""
import asyncio
import datetime
import hashlib
import operator
import os
import time
from typing import Any

import backoff
import openai
import pandas as pd
from more_itertools import chunked
from tqdm import tqdm

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.core import keychain
from src.data import prompts, rr
from src.data.evaluate import update_evaluations


TEMPLATE = prompts.load("gpt-few-shot")


def load_utterances() -> pd.DataFrame:
    return rr.load()[:200]


@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def get_completion(
    client: openai.AsyncOpenAI,
    utterance: pd.Series,
    model_name: str,
    temperature: float = 1.0
) -> dict[str, Any]:
    prompt = TEMPLATE.format(utterance=utterance.utterance)
    result = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=temperature,
    )
    return utterance.to_dict() | {
        "prompt_template": TEMPLATE,
        "prompt": prompt,
        "temperature": temperature,
        "model_name": result.model,
        "timestamp": datetime.datetime.now(),
        "generation": result.choices[0].message.content,
    }


async def get_completions(
    model_name: str, temperature: float = 1.0
) -> list[dict[str, Any]]:
    ret = []
    client = openai.AsyncOpenAI()
    us = map(operator.itemgetter(1), load_utterances().iterrows())
    tasks = [get_completion(client, u, model_name, temperature) for u in us]
    for chunk in tqdm(list(chunked(tasks, n=60))):
        ret += await asyncio.gather(*chunk)
        # Trying to respect rate limits.
        time.sleep((60 / 5000) * len(chunk) * 10)
    return ret 


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "zero-shot"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-m", "--model", default="gpt-3.5-turbo")
    ctx.parser.add_argument("-t", "--temperature", type=float, default=1.0)
    args = ctx.parser.parse_args()
    # Generate dialogues asynchronously.
    os.environ["OPENAI_API_KEY"] = keychain.get("IACS")
    completions = pd.DataFrame(asyncio.run(get_completions(args.model, args.temperature)))
    # Write generations to file.
    model_name = completions.iloc[0]["model_name"]
    phash = hashlib.shake_256(TEMPLATE.encode("utf-8")).hexdigest(8)
    outname = datetime.datetime.now().strftime(
        f"{model_name}_{phash}_%Y%m%d.%H%M%S.csv.gz"
    )
    outpath = os.path.join(args.outdir, outname)
    os.makedirs(args.outdir, exist_ok=True)
    completions.to_csv(outpath, index=False, compression="gzip")
    ctx.log.info("wrote: %s", outpath)
    update_evaluations(args.outdir)
    ctx.log.info("updated: %s", os.path.join(args.outdir, "results.csv"))


if __name__ == "__main__":
    harness(main)

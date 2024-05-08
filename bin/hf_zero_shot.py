#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run HuggingFace models zero-shot on our generated prompts.

Usage Examples:
    $ hf_zero_shot.py                   # No args needed.
    $ hf_zero_shot.py -o path/to/outdir # Custom outdir.
    $ hf_zero_shot.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import datetime
import hashlib
import operator
import os
import time
from typing import Any

import torch
import pandas as pd
import transformers as tf
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
    return rr.load()[:5]


def get_completion(
    pipeline: tf.Pipeline,
    utterance: pd.Series,
    model_name: str,
) -> dict[str, Any]:
    prompt = pipeline.tokenizer.apply_chat_template(
        [{
            "role": "user",
            "content": TEMPLATE.format(utterance=utterance.utterance)
        }],
        tokenize=False,
        add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    result = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return utterance.to_dict() | {
        "prompt_template": TEMPLATE,
        "prompt": prompt,
        "model_name": model_name,
        "timestamp": datetime.datetime.now(),
        "generation": result[0]["generated_text"],
    }


def get_completions(model_name: str) -> list[dict[str, Any]]:
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves inference speed.
    torch.compile(mode="reduce-overhead")         # Improves inference speed.
    pipeline = tf.pipeline(
        task="text-generation",
        model=model_name,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    us = list(map(operator.itemgetter(1), load_utterances().iterrows()))
    return [get_completion(pipeline, u, model_name) for u in tqdm(us)]


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "zero-shot"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-m", "--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    #  ctx.parser.add_argument("-p", "--prompt", default="gpt-zero-shot")
    args = ctx.parser.parse_args()
    # Generate dialogues asynchronously.
    completions = pd.DataFrame(get_completions(args.model))
    # Write generations to file.
    model_name = completions.iloc[0]["model_name"]
    phash = hashlib.shake_256(TEMPLATE.encode("utf-8")).hexdigest(8)
    outname = datetime.datetime.now().strftime(
        f"{model_name.replace('/', '_')}_{phash}_%Y%m%d.%H%M%S.csv.gz"
    )
    outpath = os.path.join(args.outdir, outname)
    os.makedirs(args.outdir, exist_ok=True)
    completions.to_csv(outpath, index=False, compression="gzip")
    ctx.log.info("wrote: %s", outpath)
    results = update_evaluations(args.outdir)
    ctx.log.info("updated: %s", os.path.join(args.outdir, "results.csv"))
    rlines = results.set_index("path").loc[outpath].to_string().split("\n")
    list(map(ctx.log.info, rlines))


if __name__ == "__main__":
    harness(main)

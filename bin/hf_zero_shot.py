#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run HuggingFace models zero-shot on our generated prompts.

Usage Examples:
    $ hf_zero_shot.py                   # No args needed.
    $ hf_zero_shot.py -o path/to/outdir # Custom outdir.
    $ hf_zero_shot.py --model meta-llama/Meta-Llama-3-8B-Instruct

TODO:
    * Support reading in a generation config.
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
from datasets import Dataset
from more_itertools import chunked
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.core import keychain
from src.data import prompts, rr
from src.data.evaluate import update_evaluations


TEMPLATE = None


def load_utterances(pipeline: tf.Pipeline) -> Dataset:
    utts = rr.load()

    def get_prompt(utterance: str) -> str:
        return pipeline.tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": TEMPLATE.format(utterance=utterance)
            }],
            tokenize=False,
            add_generation_prompt=True
        )

    utts = utts.assign(prompt=utts.utterance.map(get_prompt))
    return KeyDataset(Dataset.from_pandas(utts, preserve_index=False), "prompt")


def get_completions(model_name: str) -> list[dict[str, Any]]:
    ret = []
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves inference speed.
    torch.compile(mode="reduce-overhead")         # Improves inference speed.
    pipeline = tf.pipeline(
        model=model_name,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    if not getattr(pipeline.model.config, "is_encoder_decoder"):
        pipeline.tokenizer.padding_side = "left"  # For decoder-only models.
    if pipeline.tokenizer.pad_token_id is None:
        pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
    utts = load_utterances(pipeline)
    pipeline_iter = pipeline(
        utts,
        batch_size=4,
        max_new_tokens=256,
        eos_token_id=terminators,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    for rdx, result in enumerate(tqdm(pipeline_iter, total=len(utts))):
        ret.append(utts.dataset[rdx] | {
            "prompt_template": TEMPLATE,
            "model_name": model_name,
            "timestamp": datetime.datetime.now(),
            "generation": result[0]["generated_text"],
        })
    return ret


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "data", "zero-shot"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-m", "--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ctx.parser.add_argument("-p", "--prompt", default="gpt-zero-shot")
    args = ctx.parser.parse_args()
    # Set the current prompt template.
    global TEMPLATE
    TEMPLATE = prompts.load(args.prompt)
    # Generate dialogues.
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

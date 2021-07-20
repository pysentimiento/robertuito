import os
import torch
import fire
import pandas as pd
import json
from finetune_vs_scratch.sentiment import run as run_sentiment
from finetune_vs_scratch.emotion import run as run_emotion
from finetune_vs_scratch.context_hate import run as run_context_hate
from finetune_vs_scratch.hate import run as run_hateval

tasks = {
    "hate": run_hateval,
    "context_hate": run_context_hate,
    "sentiment": run_sentiment,
    "emotion": run_emotion,
}

def run_benchmark(model_name: str, times: int, output_path: str, limit: int = None, max_epochs: int = None, tasks=None):
    """
    Run benchmark

    Arguments:

    model_name:
        Model name or path. If a model name, should be present at huggingface's model hub

    times:
        Number of times

    output_path:
        Where to output results (in JSON format)

    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(("*"*80+'\n')*3)
    print(f"Running benchmark with {model_name}")
    print(f"Using {device}", "\n"*3)
    print(("*"*80+'\n')*3)

    task_args = {
        "device": device,
        "limit": limit,
    }

    if max_epochs:
        task_args["epochs"] = max_epochs

    print(task_args)

    results = {k: [] for k in tasks}
    results["model_name"] = model_name

    for i in range(times):
        print(("="*80+'\n')*3)
        print(f"{i+1} iteration", "\n"*3)
        seed = 2021 + i

        for task_name, task_fun in tasks.items():
            task_results = task_fun(model_name, seed=seed, **task_args)
            results[task_name].append(task_results)

            print(task_name)
            for k, v in task_results.items():
                print(f"{k} = {v:.4f}")

    with open(output_path, "w+") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")




if __name__ == '__main__':
    fire.Fire(run_benchmark)
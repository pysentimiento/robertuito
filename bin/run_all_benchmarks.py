import os
import re
import fire
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def run_all(times=10):
    models = [
        ("finiteautomata/robertuito-base-uncased", "robertuito-uncased.json"),
        ("finiteautomata/robertuito-base-cased", "robertuito-cased.json"),
        ("finiteautomata/robertuito-base-deacc", "robertuito-deacc.json"),
        ("bertin-project/bertin-roberta-base-spanish", "bertin.json"),
        ("BSC-TeMU/roberta-base-bne", "roberta-bne.json"),
    ]
    logger.info("Running benchmarks")
    for model_name, output_path in models:

        logger.info(f"Running model: {model_name}")

        output_path=f"output/{output_path}.json"

        cmd = f"python bin/run_benchmark.py {model_name} {times} {output_path} --max_length 128"

        os.system(cmd)


if __name__ == "__main__":
    fire.Fire(run_all)


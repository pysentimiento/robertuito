import os
import re
import fire
import json
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def run_all(times=10):
    models = [
        #("finiteautomata/robertuito-base-uncased", "robertuito-uncased.json"),
        #("finiteautomata/robertuito-base-cased", "robertuito-cased.json"),
        #("finiteautomata/robertuito-base-deacc", "robertuito-deacc.json"),
        #("bertin-project/bertin-roberta-base-spanish", "bertin.json"),
        #("BSC-TeMU/roberta-base-bne", "roberta-bne.json"),
        ("dccuchile/bert-base-spanish-wwm-uncased", "beto-uncased.json"),
        ("models/beto-uncased-2500", "beto-uncased-2500.json"),
        ("models/beto-uncased-5000", "beto-uncased-5000.json"),
        ("models/beto-uncased-10000", "beto-uncased-10000.json"),
        ("models/beto-uncased-20000", "beto-uncased-20000.json"),
        ("dccuchile/bert-base-spanish-wwm-cased", "beto-cased.json"),
        ("models/beto-cased-2500", "beto-cased-2500.json"),
        ("models/beto-cased-5000", "beto-cased-5000.json"),
        ("models/beto-cased-10000", "beto-cased-10000.json"),
        ("models/beto-cased-20000", "beto-cased-20000.json"),
    ]
    logger.info("Running benchmarks")
    for model_name, output_path in models:

        logger.info(f"Running model: {model_name}")

        output_path=f"output/{output_path}"

        if os.path.exists(output_path):
            with open(output_path) as f:
                report = json.load(f)

            run_times = len(report["hate"])
            if run_times >= times:
                logger.info(f"Skipping model: {model_name}")
                continue
            else:
                logger.info(f"Found {run_times}")
                effective_times = times - run_times
        else:
            effective_times = times

        cmd = f"python bin/run_benchmark.py {model_name} {effective_times} {output_path} --max_length 128"

        os.system(cmd)


if __name__ == "__main__":
    fire.Fire(run_all)


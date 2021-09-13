## Benchmarks

```bash
python bin/run_benchmark.py 'dccuchile/bert-base-spanish-wwm-uncased' --times 5 output/beto-uncased.json --task irony
python bin/run_benchmark.py 'dccuchile/bert-base-spanish-wwm-cased' --times 5 output/beto-cased.json --task irony
python bin/run_benchmark.py 'finiteautomata/robertuito-base-uncased' --times 5 output/robertuito-uncased-200k.json --task irony
```
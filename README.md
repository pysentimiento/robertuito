# finetune_vs_scratch

0. Install requirements

```
poetry install
```


### Smoke test

Test the benchmark running

```
./smoke_test.sh
```

### TPU Training

1. Generate arrow dataset
```
python bin/generate_dataset.py data/filtered_tweets/ data/arrow/dataset/
```
2. Train

On v2 tpu => this without generating dataset
```
num_steps=15000; python bin/xla_spawn.py --num_cores 8 bin/finetune_lm.py --input_dir data/filtered_tweets/ --output_dir models/beto-uncased-${num_steps}/ --num_steps $num_steps --model_name 'dccuchile/bert-base-spanish-wwm-uncased' --per_device_batch_size 64 --accumulation_steps 4  --num_proc 8  --on_the_fly
```
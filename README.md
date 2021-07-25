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
```
num_steps=125000; python bin/finetune_lm.py --dataset_path  data/arrow/dataset/ --output_dir models/beto-uncased-${num_steps}/ --num_steps $num_steps --model_name 'dccuchile/bert-base-spanish-wwm-uncased' --num_tpu_cores 8 --per_device_batch_size 64 --accumulation_steps 4  --on_the_fly
```
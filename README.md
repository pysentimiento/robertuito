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

### Finetuning

On v2 tpu => this without generating dataset
```
num_steps=15000; python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py --input_dir data/filtered_tweets/ --output_dir models/beto-uncased-${num_steps}/ --num_steps $num_steps --model_name 'dccuchile/bert-base-spanish-wwm-uncased' --per_device_batch_size 64 --accumulation_steps 4  --num_proc 8  --on_the_fly --finetune
```

On v3 tpu =>

```
num_proc=32
num_steps=25000; python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py --input_dir data/filtered_tweets/ --output_dir models/beto-uncased-${num_steps}/ --num_steps $num_steps --model_name 'dccuchile/bert-base-spanish-wwm-uncased' --per_device_batch_size 128 --accumulation_steps 2 --num_proc $num_proc  --on_the_fly --finetune
```

### Train from scratch

On v3 TPU

```
model="models/twerto-base-uncased"
num_proc=16 #Check your CPU cores
num_steps=3000
output_dir="models/twerto-base-uncased-${num_steps}"
python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py\
    --input_dir data/filtered_tweets/ --output_dir $output_dir --model_name $model \
    --num_steps $num_steps  --per_device_batch_size 128 --accumulation_steps 2\
    --eval_steps 500 --save_steps 2000\
    --num_proc $num_proc  --on_the_fly
```

### Academic budget

```
model="models/twerto-base-uncased"
num_proc=16 #Check your CPU cores
num_steps=30000
pdbs=128
acc=4
lr=0.002
output_dir="models/twerto-base-uncased-${num_steps}"
python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py\
    --input_dir data/filtered_tweets/ --output_dir $output_dir --model_name $model \
    --num_steps $num_steps  --per_device_batch_size $pdbs --accumulation_steps $acc\
    --learning_rate $lr\
    --eval_steps 500 --save_steps 2000\
    --num_proc $num_proc --on_the_fly
```


### GPU
Con deepspeed

```
model="models/twerto-base-uncased"
num_proc=24 #Check your CPU cores
num_steps=10
batch_size=32
acc=64
output_dir="models/twerto-base-uncased-${num_steps}"
deepspeed --num_gpus 2 bin/run_mlm.py\
    --input_dir data/filtered_tweets/ --output_dir $output_dir --model_name $model \
    --num_steps $num_steps  --per_device_batch_size $batch_size --accumulation_steps $acc\
    --eval_steps 500 --save_steps 2000\
    --num_proc $num_proc --on_the_fly
```

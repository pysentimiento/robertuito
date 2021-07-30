# finetune_vs_scratch


## Setup

### Finetuning


On v2 tpu => this without generating dataset
```

export TPU_IP_ADDRESS=10.97.22.154
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
pdbs=64
acc=4
lr=0.0006
num_proc=16
num_steps=15000

output_dir="models/beto-uncased-${num_steps}/"
python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py\
    --input_dir data/filtered_tweets/ --output_dir $output_dir --num_steps $num_steps\
    --model_name 'dccuchile/bert-base-spanish-wwm-uncased'\
    --per_device_batch_size $pdbs --accumulation_steps 4\
    --num_proc $num_proc --finetune
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
```bash
export TPU_IP_ADDRESS=10.110.227.66
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

model="models/twerto-base-uncased"
num_proc=16 #Check your CPU cores
num_steps=500000
pdbs=128
acc=4
lr=0.0006
output_dir="models/twerto-base-uncased-${num_steps}"
python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py\
    --input_dir data/filtered_tweets/ --output_dir $output_dir --model_name $model \
    --num_steps $num_steps  --per_device_batch_size $pdbs --accumulation_steps $acc\
    --learning_rate $lr --on_the_fly\
    --eval_steps 500 --save_steps 2000\
    --num_proc $num_proc
    #--resume_from_checkpoint
```


## run_mlm_hf.py

```bash
python bin/xla_spawn.py --num_cores 8 bin/run_mlm_hf.py\
    --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 0.000001\
    --eval_steps 1000 --save_steps 3000 --logging_steps 50\
    --per_device_batch_size
    --input_dir data/filtered_tweets/ --output_dir $output_dir --model_name $model \
    --num_steps $num_steps  --per_device_batch_size $pdbs --accumulation_steps $acc\
    --learning_rate $lr --on_the_fly\
    --eval_steps 500 --save_steps 2000\
    --num_proc $num_proc
```

### GPU
Con deepspeed

```bash
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
```
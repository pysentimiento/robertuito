0. Tmux y sarasa

```
echo 'set -g default-terminal "xterm-256color"' >> .tmux.conf
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global user.email jmperez.85@gmail.com
git config --global user.name "Juan Manuel Pérez"
 git config --global core.editor vi
```

1. Install pyenv

Run
```
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

pyenv install 3.8.11

```

2. Create TPUs

```
gcloud config set compute/zone us-central1-f

tpu_name="pysentimiento-tpu-ne"
echo "Creating ${tpu_name}"
gcloud compute tpus create $tpu_name \
    --accelerator-type=v2-8 \
    --version=pytorch-1.9 \
    --preemptible
```

3.

```
export TPU_IP_ADDRESS=XXXXXXX
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

4. Clonar repo

```
mkdir projects && cd projects
git clone git
cd finetune_vs_scratch


5. Copiar datos

```
gsutil cp -r gs://pysentimiento/filtered_tweets data/
```

6. Correr finetuning

```
num_steps=12500; python bin/finetune_lm.py --dataset_path  data/arrow/dataset/ --output_dir models/beto-uncased-${num_steps}/ --num_steps $num_steps --model_name 'dccuchile/bert-base-spanish-wwm-uncased' --num_tpu_cores 8 --per_device_
batch_size 64 --accumulation_steps 4  --num_proc 8 --on_the_fly
```
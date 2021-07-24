0. Tmux y sarasa

```
echo 'set -g default-terminal "xterm-256color"' >> .tmux.conf
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
for i in $(seq -f "%03g" 1 5)
do
    tpu_name="pysentimiento-tpu-${i}"
    echo "Creating ${tpu_name}"
    gcloud compute tpus create $tpu_name \
        --accelerator-type=v2-8 \
        --version=pytorch-1.9 \
        --preemptible
done
```

O crear una sola
```
gcloud config set compute/zone us-central1-f
tpu_name="pysentimiento-tpu"
echo "Creating ${tpu_name}"
gcloud compute tpus create $tpu_name \
    --accelerator-type=v2-8 \
    --version=pytorch-1.9 \
    --preemptible
```

3.

```
export TPU_IP_ADDRESS=TPU_IP
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

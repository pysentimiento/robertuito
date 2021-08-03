0. Create instance


## v2 instance

```bash
NAME="saturno-cased"
gcloud compute instances create $NAME \
    --boot-disk-size=200GB \
    --boot-disk-type="pd-balanced" \
    --machine-type "e2-standard-8" \
    --image-family="pytorch-1-9-xla-debian-10" \
    --image-project=ml-images  \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --zone us-central1-f \
    --preemptible
```

## v3 instance

```
NAME="gaia"
gcloud compute instances create $NAME \
    --boot-disk-size=200GB \
    --boot-disk-type="pd-balanced" \
    --machine-type "e2-standard-16" \
    --image-family="pytorch-1-9-xla-debian-10" \
    --image-project=ml-images  \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --zone europe-west4-a \
    --preemptible
```

1. Tmux y sarasa

```bash
sudo rm /var/lib/apt/lists/lock # Por alguna razón está lockeado
echo 'set -g default-terminal "xterm-256color"' >> .tmux.conf
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global user.email jmperez.85@gmail.com
git config --global user.name "Juan Manuel Pérez"
git config --global core.editor vi
```



2. Create TPUs

```
gcloud config set compute/zone us-central1-f

tpu_name="pysentimiento-tpu-2"
echo "Creating ${tpu_name}"
gcloud compute tpus create $tpu_name \
    --accelerator-type=v2-8 \
    --version=pytorch-1.9 \
    --preemptible
```

### v3

```
gcloud config set compute/zone europe-west4-a

tpu_name="gaia-tpu"
echo "Creating ${tpu_name}"
gcloud compute tpus create $tpu_name \
    --accelerator-type=v3-8 \
    --version=pytorch-1.9
```


3. Configurar

```
export TPU_IP_ADDRESS=10.97.22.154
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

4. Clonar repo

```
mkdir projects && cd projects
git clone git@github.com:finiteautomata/finetune_vs_scratch.git
cd finetune_vs_scratch
pip install poetry
poetry install
poetry run pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl # Chequear que esto esté ok

```


5. Copiar datos

```
mkdir data
gsutil -m cp -r gs://pysentimiento/filtered_tweets data/
cd data && mv filtered_tweets tweets
cd tweets && mkdir train && mkdir test
mv spanish-tweets-099.txt test/
mv *.txt train/
```

## Correr finetuning

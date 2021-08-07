0. Create instance

Link precios:
1. https://cloud.google.com/products/calculator/#id=e89ffef9-b07b-4379-8d65-3eee6508c814
2. https://cloud.google.com/products/calculator/#id=1285205d-f417-4dae-81d7-10633ba291f8

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
gcloud compute instances create $NAME  \
    --boot-disk-size=160GB \
    --boot-disk-type="pd-balanced" \
    --machine-type "e2-custom-32-50176" \
    --image-family="pytorch-1-9-xla-debian-10" \
    --image-project=ml-images \
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

cat >> .bashrc

export TPU_IP_ADDRESS=10.110.227.66
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
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

tpu_name="gaia-tpu-2"
echo "Creating ${tpu_name}"
gcloud compute tpus create $tpu_name \
    --accelerator-type=v3-8 \
    --version=pytorch-1.9
```



3. Clonar repo

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
gsutil -m cp gs://pysentimiento/data .
```

6. Crear swap...

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Correr finetuning

### COLAB

1. Go to `SSH` colab
2. Connect to SSH instance
3. Run
```bash
cd /content/finetune_vs_scratch
python bin/xla_spawn.py --num_cores 8 bin/run_mlm.py /content/drive/MyDrive/models/scratch-uncased-colab.json
```

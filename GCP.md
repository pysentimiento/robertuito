0. Tmux y sarasa

```
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

tpu_name="pysentimiento-tpu-ne"
echo "Creating ${tpu_name}"
gcloud compute tpus create $tpu_name \
    --accelerator-type=v2-8 \
    --version=pytorch-1.9 \
    --preemptible
```

3. Configurar

```
export TPU_IP_ADDRESS=XXXXXXX
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

4. Clonar repo

```
mkdir projects && cd projects
git clone git@github.com:finiteautomata/finetune_vs_scratch.git
cd finetune_vs_scratch
pip install .
```


5. Copiar datos

```
gsutil cp -r gs://pysentimiento/filtered_tweets data/
```

6. Correr finetuning

```
num_steps=12500; python bin/finetune_lm.py --dataset_path  data/arrow/dataset/ --output_dir models/beto-uncased-${num_steps}/ --num_steps $num_steps --model_name 'dccuchile/bert-base-spanish-wwm-uncased' --num_tpu_cores 8 --per_device_
batch_size 64 --accumulation_steps 4  --num_proc 8 --on_the_fly
```
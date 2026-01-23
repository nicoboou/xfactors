#!/bin/bash

cd /projects/compures/alexandre/disdiff_adapters
source /projects/compures/alexandre/.venv/bin/activate

batch_size=32
max_epochs=5
dataset="shapes"
betas=("1")
latent_dim=64
warm_up="False"
lr=1e-05
arch="res"
version_model="vae"
gpus="1"


echo $dataset
for beta in $betas
do

    experience="batch${batch_size}/test_dim${latent_dim}"

    log_dir="disdiff_adapters/logs/${version_model}/${dataset}/${experience}/vae_epoch=${max_epochs}_beta=${beta}_latent=${latent_dim}_warm_up=${warm_up}_lr=${lr}_batch=${batch_size}_arch=${arch}"
    mkdir -p $log_dir
    python3 -m disdiff_adapters.arch.vae.train \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                --lr $lr \
                                --arch $arch \
                                --gpus $gpus
                                # 2>&1 | tee "${log_dir}/log.out" 

    python3 -m disdiff_adapters.arch.vae.test \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta $beta \
                                --latent_dim $latent_dim \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                --lr $lr \
                                --arch $arch \
                                --gpus $gpus

done
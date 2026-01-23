#!/bin/bash

cd /projects/compures/alexandre/disdiff_adapters
source /projects/compures/alexandre/.venv/bin/activate

batch_size=32
max_epochs=500
dataset="cars3d"
beta_s="$1"
beta_t=1.0
latent_dims_s=("126")
latent_dim_t=2
warm_up="False"
lr=1e-05
arch="res"
loss_type="$2"
l_cov=0.0
l_nce="0.1"
l_anti_nce=0.0
key="factor=2"
version_model="$3"
gpus="${4:-0}"

echo $dataset

for latent_dim_s in $latent_dims_s; do

    experience="test_factor_2/batch${batch_size}/test_dim_s${latent_dim_s}"
    log_dir="disdiff_adapters/logs/${version_model}/${dataset}/loss_${loss_type}/${experience}/md_epoch=${max_epochs}_beta=(${beta_s},${beta_t})_latent=(${latent_dim_s},${latent_dim_t})_batch=${batch_size}_warm_up=${warm_up}_lr=${lr}_arch=${arch}+l_cov=${l_cov}+l_nce=${l_nce}+l_anti_nce=${l_anti_nce}_${key}"
    mkdir -p $log_dir

    python3 -m disdiff_adapters.arch.multi_distillme.train \
            --max_epochs $max_epochs \
            --dataset $dataset \
            --beta_s $beta_s \
            --beta_t $beta_t \
            --latent_dim_s $latent_dim_s \
            --latent_dim_t $latent_dim_t \
            --batch_size $batch_size \
            --warm_up $warm_up \
            --lr $lr \
            --arch $arch \
            --gpus $gpus \
            --key $key \
            --loss_type $loss_type \
            --l_cov $l_cov \
            --l_nce $l_nce \
            --l_anti_nce $l_anti_nce \
            --experience $experience \
            --version_model $version_model \
            2>&1 | tee "${log_dir}/log.out"

    exitcode=${PIPESTATUS[0]}
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EXIT CODE = ${exitcode}" >> "${log_dir}/log.out"

done

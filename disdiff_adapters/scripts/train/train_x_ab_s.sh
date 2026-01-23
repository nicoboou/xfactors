#!/bin/bash

cd /projects/compures/alexandre/disdiff_adapters
source /projects/compures/alexandre/.venv/bin/activate

batch_size=32
max_epochs=70
dataset="mpi3d"
beta_t=1.0
latent_dims_s=("126")
dims_by_factors="2"
select_factors="-1"
warm_up="False"
lr=1e-05
arch="res"
loss_type="vae_nce"
l_cov=0.0
l_nce_by_factors="0.1"
l_anti_nce=0.0
key="factor=-1"
version_model="abblation/x_ab_s_with_beta_t1"

gpus="0"

echo $dataset

for latent_dim_s in $latent_dims_s; do
    experience="factor${select_factors}/batch${batch_size}"

    log_dir="disdiff_adapters/logs/${version_model}/${dataset}/loss_${loss_type}/${experience}/x_ab_s_epoch=${max_epochs}_beta=(${beta_t})_latent=(${dims_by_factors})_batch=${batch_size}_warm_up=${warm_up}_lr=${lr}_arch=${arch}+l_cov=${l_cov}+l_nce=${l_nce_by_factors}+l_anti_nce=${l_anti_nce}_${key}"
    mkdir -p $log_dir

    python3 -m disdiff_adapters.arch.multi_distillme.train_x_ab_s \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta_t $beta_t \
                                --dims_by_factors $dims_by_factors \
                                --batch_size $batch_size \
                                --warm_up $warm_up \
                                --lr $lr \
                                --arch $arch \
                                --gpus $gpus \
                                --key $key \
                                --loss_type $loss_type \
                                --l_cov $l_cov \
                                --l_nce_by_factors $l_nce_by_factors \
                                --l_anti_nce $l_anti_nce \
                                --experience $experience \
                                --version_model $version_model \
                                2>&1 | tee "${log_dir}/log.out" 

    exitcode=${PIPESTATUS[0]}
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EXIT CODE = ${exitcode}" >> "${log_dir}/log.out"
done
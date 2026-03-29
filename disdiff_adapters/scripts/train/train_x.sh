#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2 bash disdiff_adapters/scripts/train/train_x.sh 1.0 all shapes_downsample 0,1,2 40

cd /projects/compures/nicolas/xfactors
source /projects/compures/nicolas/xfactors/.venv/bin/activate

batch_size=32
max_epochs=70
dataset="shapes"
beta_s="$1"
beta_t=100.0
latent_dims_s=("126")
dims_by_factors="2"
select_factors="${5:-}" # "0,2,6"
warm_up="False"
lr=1e-05
arch="res"
loss_type="$2"
l_cov=0.0
l_nce_by_factors="0.1"
l_anti_nce=0.0
key="_with_merge"
version_model="$3"
degradation_types="inpainting" # ["none", "combo", "bilinear", "bicubic", "nearest_neighbor", "blur", "noise", "jpeg", "inpainting"]
degradation_levels="0,1"

gpus="${4:-0}"

if [ -n "$select_factors" ]; then
    select_factors_tag="$select_factors"
    select_factors_arg=(--select_factors "$select_factors")
else
    select_factors_tag="all"
    select_factors_arg=()
fi

echo $dataset

for latent_dim_s in $latent_dims_s; do
    experience="factor_${select_factors_tag}/batch${batch_size}/test_dim_s${latent_dim_s}"

    log_dir="disdiff_adapters/logs/${version_model}/${dataset}/loss_${loss_type}/${experience}/x_epoch=${max_epochs}_beta=(${beta_s},${beta_t})_latent=(${latent_dim_s},${dims_by_factors})_batch=${batch_size}_warm_up=${warm_up}_lr=${lr}_arch=${arch}+l_cov=${l_cov}+l_nce=${l_nce_by_factors}+l_anti_nce=${l_anti_nce}_${key}"
    mkdir -p $log_dir

    python3 -m disdiff_adapters.arch.multi_distillme.train_x \
                                --max_epochs $max_epochs \
                                --dataset $dataset \
                                --beta_s $beta_s \
                                --beta_t $beta_t \
                                --latent_dim_s $latent_dim_s \
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
                                --degradation_types $degradation_types \
                                --degradation_levels $degradation_levels \
                                "${select_factors_arg[@]}"

    exitcode=$?
    echo "$(date '+%Y-%m-%d %H:%M:%S') - EXIT CODE = ${exitcode}" >> "${log_dir}/log.out"
done
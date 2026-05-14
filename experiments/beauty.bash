#!/usr/bin/env bash
set -euo pipefail

gpu_id=0
dataset="beauty"
model_name="llmisr_sasrec"
seed_list=(42 43 44) 
ts_user=9
ts_item=4
lambda1=0.05
cluster_moe_eta=1.4
moe_gamma=1.7
moe_dyn_source="both"
moe_conf_thresh=0.0
moe_broadcast_min_len=0
fuse_len_gamma=0.0
alpha=0.001
beta=0.05

# =============================================================================

for seed in "${seed_list[@]}"; do
    
    echo ">>> Running Seed: ${seed} ..."
    
    python main.py --dataset ${dataset} \
        --model_name ${model_name} \
        --gpu_id ${gpu_id} \
        --seed ${seed} \
        --check_path "seed${seed}" \
        --ts_user ${ts_user} \
        --ts_item ${ts_item} \
        --freeze \
        --log \
        --use_cross_att \
        --lambda1 ${lambda1} \
        --cluster_moe_eta ${cluster_moe_eta} \
        --moe_gamma ${moe_gamma} \
        --moe_dyn_source ${moe_dyn_source} \
        --moe_conf_thresh ${moe_conf_thresh} \
        --moe_broadcast_min_len ${moe_broadcast_min_len} \
        --fuse_len_gamma ${fuse_len_gamma} \
        --alpha ${alpha} \
        --beta ${beta} 
        
    echo ">>> Seed ${seed} completed."
    echo "----------------------------------------------------------"

done

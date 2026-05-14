#!/usr/bin/env bash
set -euo pipefail

gpu_id=0
dataset="fashion"
model_name="llmisr_sasrec"
seed_list=(42 43 44)  
ts_user=3
ts_item=4
lambda1=0.1
cluster_moe_eta=0.6
moe_gamma=1.0
moe_dyn_source="seq_last"
moe_conf_thresh=0.05
moe_broadcast_min_len=5
fuse_len_gamma=0.5     
alpha=0.001 
beta=0.01  
top1_margin_weight=0.05
score_mix=0.5


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
        --beta ${beta} \
        --top1_margin_weight ${top1_margin_weight} \
        --score_mix ${score_mix} 
        
    echo ">>> Seed ${seed} completed."
    echo "----------------------------------------------------------"

done

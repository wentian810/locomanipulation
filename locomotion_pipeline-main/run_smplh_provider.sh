# --target_dir data/repair/repaired_zitai_batch1 \
# --target_dir data/storage/pkl_struct/batch1/zitai/zitai_n10000_idx002/motionx \
# --target_dir output/batch1_zitai_250-all_check/results/152225_6_min_Intense_Lower_Abs04 \

python code/optim/smplh_provider.py \
    --target_dir /workspace/projects_dataset/data_deliver/smpl2qiaojie/data/storage/pkl_struct/batch1/zitai/zitai_n10000_idx002/motionx/438061_HOURGLASS_WORKOUT_for41 \
    --file_pattern '*/smpl*.npz' \
    --gravity_axis 'y-' \
    --seqID_index 2 \
    --run_optim 0 \
    --run_detect_penetration 1 \
    --check_gravity 0 \
    --save_dir output/vis/vis3d

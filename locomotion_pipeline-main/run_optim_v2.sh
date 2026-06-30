# python code/optim/optimizer_v2.py \
#     --config code/configs/config.yaml \
#     --file_pattern "*/*.npz" \
#     --use_timestamp 0

python code/optim/optimizer_v2.py \
    --config code/configs/config.yaml \
    --target_dir output_submit/batch1/zitai/zitai_n10000_idx000 \
    --file_pattern "*/*.npz" \
    --seqID_index 1 \
    --prefix batch1_test_run_speed \
    --use_timestamp 0 \
    --optim_height 1 \
    --gravity_axis 'y-' \
    --check_penetration 1 \
    --check_speed 1 \
    --gravity_alignment 1

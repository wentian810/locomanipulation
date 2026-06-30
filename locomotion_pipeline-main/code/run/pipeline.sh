# Run the pipeline

# path related
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd $PROJECT_ROOT
echo "Project root directory: $PROJECT_ROOT"

# 1. download the files from remote to local
# 1.1 download video files
bash code/scripts/run_fsys_sync.sh \
    --remote-path "syna/synadata-source-wlcb/datasets/shenbipai" \
    --local-root "${PROJECT_ROOT}/data/storage/input_struct" \
    --regex-pattern "batch1/姿态转换/321701_TIGHT_ABS_WAIST___6_min_433/321701_TIGHT_ABS_WAIST___6_min04.mp4" \
    --mode "dl" \
    --max-workers 4

# # 1.2 download pkl files
# bash code/scripts/run_fsys_sync.sh \
#     --remote-path "syna/synadata-source-wlcb/output/shenbipai" \
#     --local-root "${PROJECT_ROOT}/data/storage/pkl_struct" \
#     --regex-pattern "data/storage/pkl_struct/batch1/zitai/zitai_n10000_idx000/motionx/*.zip" \
#     --mode "dl" \
#     --max-workers 4

# # 3. unzip the downloaded files
# bash code/scripts/unzip_files.sh \
#     "data/storage/pkl_struct/batch1/zitai/zitai_n10000_idx000/motionx"

# # 4. run optimization
# python code/optim/optimizer_v2.py \
#     --config code/configs/config.yaml \
#     --target_dir output_submit/batch1/zitai/zitai_n10000_idx000 \
#     --file_pattern "*/*.npz" \
#     --seqID_index 1 \
#     --prefix batch1_test_run_speed \
#     --use_timestamp 0 \
#     --optim_height 1 \
#     --gravity_axis 'y-' \
#     --check_penetration 1 \
#     --check_speed 1 \
#     --gravity_alignment 1

# # 5. zip output files
# bash code/scripts/zip_bash.sh \
#     "output_submit/batch1/zitai/zitai_n10000_idx000" \
#     --remove-source 1

# # else. 剩下的问题是，要能够把给定一个 zip 文件， 找到其在远程路径中的位置， 以便下载其对应的 mp4.

import os
import re
import sys

pwd = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.abspath(os.path.join(pwd, '..'))
if not code_root in sys.path:
    sys.path.insert(0, code_root)

from tqdm import tqdm
import glob
from smplh_provider import Smplh_sequence
from utils.statics import SequenceStatistics
from utils.log_utils import MyLogger
from configs.config import parse_args, CfgParser

def prepare_sequences(cfg_parser, logger):
    target_dir = cfg_parser['input.target_dir']
    file_pattern = cfg_parser['input.file_pattern']
    seqID_index = cfg_parser['input.seqID_index']

    # Find all files matching the pattern
    search_pattern = os.path.join(target_dir, file_pattern)
    all_files = glob.glob(search_pattern)
    all_files = sorted(all_files)  # Sort files for consistent ordering

    num_files = len(all_files)
    logger.message(f"Found {num_files} files matching pattern: {search_pattern}")

    # Group files by sequence ID
    sequences_list = []
    for file_path in all_files:
        fname_ext = os.path.basename(file_path)
        dot_index = fname_ext.rfind('.')
        fname, ext = fname_ext[:dot_index], fname_ext[dot_index:]
        
        if ext not in ['.npz', '.pkl']:
            logger.info(f'Skipping unsupported file: {file_path}')
            continue

        if seqID_index > 0:
            tmp_path = file_path
            for _ in range(seqID_index):
                tmp_path = os.path.dirname(tmp_path)
            seq_id = os.path.basename(tmp_path)
        else:
            seq_id = fname  # 如果没有指定索引，就直接用文件名（不带扩展）作为序列ID
        sequences_list.append((seq_id, file_path))

    return sequences_list

def process_sequence(
    seq_id: str, 
    file_path: str,
    cfg_parser: CfgParser, 
    logger: MyLogger, 
):
    output_results_dir = cfg_parser['output.structure.concrete_dir.results_filter']

    smplh_seq = Smplh_sequence(file_path, logger=logger)
    seq_stats = SequenceStatistics()
    output_seq_dir = os.path.join(output_results_dir, seq_id)

    # check if optimized result already exists
    optimized_result_path = os.path.join(output_seq_dir, f"{seq_id}_optimized.npz")
    record_csv_path = os.path.join(output_seq_dir, cfg_parser['output.fnames.record_csv'])
    if os.path.exists(optimized_result_path) \
        and not cfg_parser['input.overwrite']:
        logger.info(f"Optimized result already exists for sequence {seq_id}, skipping optimization.")

        # load statistics from csv
        # ! 存疑， 有可能前后两次运行的时候合格标准不一样，所以会有风险
        # seq_stats.load_from_csv(record_csv_path)
        # seq_stats_all.load_from_csv(record_csv_path)
        return None, seq_stats
    else:
        logger.info(f"Processing sequence {seq_id} from file: {file_path}")
    
    check_result = {}
    validated_all = True
    
    # height optimization
    if cfg_parser['optimization.gravity.enabled']:
        optim_height_info = smplh_seq.optimize_height(
            **cfg_parser['optimization.gravity.params'], 
        )
        # 不是 enable 就先跳过， 因为考虑到可能
        seq_stats.add(
            seq_id, 
            smplh_seq.num_frames, 
            smplh_seq.num_frames - optim_height_info['out_of_range_count'], 
        )
        logger.info(f'Height optimization: out-of-range frames: {optim_height_info["out_of_range_count"]}/{smplh_seq.num_frames}')

    # TODO check penetration
    if cfg_parser['check.penetration.enabled']:
        penetration_info = smplh_seq.check_penetration(
            **cfg_parser['check.penetration.params'],
        )
        penetration_validation = not penetration_info['penetration_mask'].any() # 只要有一个穿模点就算不合格
        check_result['penetration'] = {
            'validation': penetration_validation,
            'details': penetration_info,
        }
        validated_all = validated_all and check_result['penetration']['validation']

        # 计算穿模的帧数
        penetration_frames = (penetration_info['penetration_mask'].any(axis=(1,2))) # (T,) bool array indicating which frames have any penetration
        penetration_frame_count = penetration_frames.sum()
        logger.info(f'[validation of penetration]:\t {penetration_validation}, frames with penetration: {penetration_frame_count}/{smplh_seq.num_frames}')

    # TODO check acceleration
    if cfg_parser['check.speed.enabled']:
        acceleration_info = smplh_seq.check_speed(
            **cfg_parser['check.speed.params'],
        )
        speed_validation = True
        for speed_dict in acceleration_info.values():
            for property_name, outlier_num in speed_dict.items():
                if outlier_num > 0:
                    speed_validation = False
                    break
        check_result['speed'] = {
            'validation': speed_validation,
            'details': acceleration_info,
        }
        validated_all = validated_all and speed_validation

        # 计算速度异常的帧数
        logger.info(f"[validation of speed]: \t{speed_validation}, linear_acceleration outlier frames: {acceleration_info['linear']['acceleration']}/{smplh_seq.num_frames}")

    # TODO gravity alignment
    if cfg_parser['check.gravity_alignment.enabled']:
        gravity_alignment_info = smplh_seq.check_gravity_center(
            **cfg_parser['check.gravity_alignment.params'],
        )
        gravity_validation = (gravity_alignment_info['support_ratio'] == 1.0)
        check_result['gravity_alignment'] = {
            'validation': gravity_validation,
            'details': gravity_alignment_info,
        }
        validated_all = validated_all and gravity_validation

        not_support_frames = smplh_seq.num_frames - gravity_alignment_info['in_support_num']
        logger.info(f"[validation of gravity alignment]: \t{gravity_validation}, not-support frames: {not_support_frames}/{smplh_seq.num_frames}")
    
    # save optimized result
    if len(check_result) > 0:
        if validated_all and \
                optim_height_info['out_of_range_count'] == 0:
            os.makedirs(output_seq_dir, exist_ok=True)
            smplh_seq.export_parameters(optimized_result_path)
            logger.info(f"Saved optimized result for sequence {seq_id} to {optimized_result_path}")
        else:
            logger.message(f"Sequence {seq_id} did not pass validation checks, optimized result not saved.")
    else:
        os.makedirs(output_seq_dir, exist_ok=True)
        smplh_seq.export_parameters(optimized_result_path)
        logger.info(f"Saved optimized result (no checks) for sequence {seq_id}")

    return check_result, seq_stats

def main(
    cfg_parser: CfgParser, 
    logger: MyLogger,
):
    sequences_list = prepare_sequences(cfg_parser, logger)
    seq_stats_all = SequenceStatistics()
    # output_dir = cfg_parser['output.output_dir']

    # Process each sequence
    for seq_idx, (seq_id, file_path) in tqdm(
        enumerate(sequences_list), 
        desc="Processing sequences",
        total=len(sequences_list),
    ):
        # #* 都打印的话日志会非常大，先注释掉了
        # logger.message(f"Processing sequence: {seq_id} from file: {file_path}")

        # single sequence optimization
        check_result, seq_stats = process_sequence(seq_id, file_path, cfg_parser, logger)
        seq_stats_all.merge_from_other(seq_stats)
        # if seq_idx >= 9:
        #     break

if __name__ == "__main__":
    cfg_path, cfg_parser, logger = parse_args()

    logger.message(f"Loaded config from {cfg_path}")
    logger.message(f"Output directory: {cfg_parser['output.output_dir']}")
    main(
        cfg_parser,
        logger,
    )

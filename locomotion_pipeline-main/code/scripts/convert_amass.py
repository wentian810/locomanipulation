"""
Convert AMASS-format npz to the format expected by the locomotion pipeline.

AMASS npz:
    poses:   (T, 156)  52 joints x 3
    trans:   (T, 3)
    betas:   (16,)
    gender:  ()
    mocap_framerate: ()

Pipeline npz:
    root_orient:    (T, 3)    global orientation
    pose_body:      (T, 63)   21 body joints x 3
    poses:          (T, 24, 3)  first 24 joints reshaped
    trans:          (T, 3)
    trans_original: (T, 3)
    betas:          (16,)
    gender:         ()
    mocap_frame_rate: ()
"""

import os, sys, glob, argparse
import numpy as np
from tqdm import tqdm


def convert_single(src_path, dst_path):
    src = np.load(src_path, allow_pickle=True)

    # Gender is stored as bytes in AMASS, decode to string
    gender = src['gender']
    if isinstance(gender, np.ndarray) and gender.dtype.kind == 'U':
        gender = str(gender)
    elif isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    else:
        gender = str(gender).lower()

    poses = src['poses']       # (T, 156)
    trans = src['trans']       # (T, 3)
    betas = src['betas']       # (16,)
    fps = float(src['mocap_framerate'])

    # AMASS stores all 52 joints in one flat array
    # Joint 0 = global_orient (3), Joints 1-21 = body (63), Joints 22-51 = hands
    root_orient = poses[:, :3]           # (T, 3)
    pose_body = poses[:, 3:66]           # (T, 63) — 21 body joints
    body_joints = poses[:, :72].reshape(-1, 24, 3)  # (T, 24, 3) — first 24 joints

    save_dict = {
        'poses':            body_joints,         # (T, 24, 3)
        'pose_body':        pose_body,           # (T, 63)
        'root_orient':      root_orient,         # (T, 3)
        'trans':            trans,               # (T, 3)
        'trans_original':   trans.copy(),        # (T, 3)
        'betas':            betas,               # (16,)
        'gender':           gender,              # string like 'neutral'
        'mocap_frame_rate': fps,
    }

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.savez(dst_path, **save_dict)
    return save_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True,
                        help='AMASS data root (e.g. assets/ACCAD)')
    parser.add_argument('--dst_dir', type=str, required=True,
                        help='Output directory for converted npz files')
    parser.add_argument('--pattern', type=str, default='*/*_poses.npz')
    args = parser.parse_args()

    search = os.path.join(args.src_dir, args.pattern)
    files = sorted(glob.glob(search))
    print(f'Found {len(files)} files')

    for src_path in tqdm(files):
        rel = os.path.relpath(src_path, args.src_dir)
        dst_path = os.path.join(args.dst_dir, rel)
        convert_single(src_path, dst_path)

    print(f'Done. Converted files saved to: {args.dst_dir}')


if __name__ == '__main__':
    main()

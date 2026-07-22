import os
import shutil
import gc
import sys
import numpy as np
import cv2
import torch
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from omegaconf import open_dict
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, quaternion_to_matrix
import subprocess
from torch.utils.data import DataLoader, ConcatDataset

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_coco17_skeleton_batch, draw_coco133_skeleton_batch, draw_bbx_xyxy_on_image_batch_multiperson

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseWholebodyExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy_batch, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import axis_angle_to_mat3x3, compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from einops import einsum, rearrange

from hmr4d.utils.datasets.vitdet_dataset import ViTDetDataset, recursive_to

_load_hamer = None
DEFAULT_CHECKPOINT_HAMER = None
_HAMER_IMPORT_ERROR = None
try:
    from hamer.models import load_hamer as _load_hamer, DEFAULT_CHECKPOINT_HAMER as DEFAULT_CHECKPOINT_HAMER
except ImportError as exc:
    try:
        from hamer.models import load_hamer as _load_hamer, DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HAMER
    except ImportError as fallback_exc:
        _HAMER_IMPORT_ERROR = fallback_exc or exc


CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def release_torch_memory(label=None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # NOTE: ipc_collect() is deliberately NOT called here — it can hang
        # the GPU driver when no CUDA IPC resources exist.  In a single-process
        # pipeline like ours, IPC cleanup is never needed.
    if label:
        Log.info(label)


def get_video_fps(video_path):
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    return fps


def hydra_quote(value):
    text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_args_to_cfg():
    # Put all args to cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="inputs/demo/dance_3.mp4")
    parser.add_argument("--video_name", type=str, default=None, help="by default to video.stem")
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for VitPose and VitFeat")
    parser.add_argument("--recreate_video", action="store_true", help="If true, encode the original video to 30 fps for visualization")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument("--export_pt", action="store_true", help="If true, export pt files")
    parser.add_argument("--skip_render", action="store_true", help="If true, skip rendering")
    parser.add_argument("--low_memory", action="store_true", help="Release large hand-stage objects before HMR2/GVHMR stages")
    parser.add_argument("--force_hand_preprocess", action="store_true", help="Recompute wholebody VitPose and hand MANO params")
    parser.add_argument("--hand_preprocess_only", action="store_true", help="Stop after writing VitPose wholebody and hand MANO params")
    parser.add_argument("--render_only", action="store_true", help="Render existing HMR4D results without rerunning preprocessing or body inference")
    parser.add_argument("--mano_params_override", type=str, default="", help="Use this filtered mano_params.pt for rendering")
    parser.add_argument("--hand_backend", choices=["hamer", "wilor", "hand4wholepp"], default="hamer", help="Hand reconstruction backend")
    parser.add_argument("--wilor_root", type=str, default="", help="Path to a WiLoR checkout; defaults to third-party/WiLoR")
    parser.add_argument("--wilor_checkpoint", type=str, default="", help="Path to wilor_final.ckpt")
    parser.add_argument("--wilor_config", type=str, default="", help="Path to WiLoR model_config.yaml")
    parser.add_argument("--wilor_fast", action="store_true", help="Use WiLoR fast settings when available")
    parser.add_argument("--hand4wholepp_root", type=str, default="", help="Path to a Hand4Whole++ checkout; defaults to third-party/Hand4Whole-plus-plus_RELEASE")
    parser.add_argument("--hand4wholepp_snapshot", type=str, default="", help="Path to Hand4Whole++ snapshot_6.pth")
    parser.add_argument("--hand4wholepp_python", type=str, default="", help="Python executable for the isolated Hand4Whole++ runner")
    parser.add_argument("--hand4wholepp_batch_size", type=int, default=4, help="Hand4Whole++ video inference batch size with automatic CUDA OOM fallback")
    parser.add_argument("--hand4wholepp_yolo_model", type=str, default="yolo11n.pt", help="Fallback YOLO model when GVHMR bboxes are absent")
    parser.add_argument(
        "--hand4wholepp_joint_source",
        choices=["direct_mano", "fused_smplx"],
        default="direct_mano",
        help="Export WiLoR-native MANO joints or legacy Hand4Whole++ fused SMPL-X joints",
    )
    parser.add_argument(
        "--hand4wholepp_crop_tracking",
        choices=["off", "flow_kalman"],
        default="off",
        help="Optional H4W++ DWPose hand-crop tracking; default off preserves frame-independent crops.",
    )
    parser.add_argument(
        "--hand4wholepp_crop_tracking_max_gap",
        type=int,
        default=8,
        help="Maximum DWPose-missing frames that H4W++ crop tracking may propagate.",
    )
    parser.add_argument(
        "--hand4wholepp_crop_tracking_max_prediction_gap",
        type=int,
        default=2,
        help="Maximum no-flow prediction frames inside an H4W++ crop tracking gap.",
    )
    parser.add_argument(
        "--hand4wholepp_crop_tracking_direct_observation_quality",
        type=float,
        default=0.75,
        help="High-quality DWPose observations passed directly to H4W++ without smoothing.",
    )
    parser.add_argument("--vitpose_img_ds", type=float, default=0.5, help="Image downsample factor for wholebody VitPose crops")
    parser.add_argument("--hand_kpt_conf_thr", type=float, default=0.5, help="Wholebody hand keypoint confidence threshold")
    parser.add_argument("--hand_kpt_low_conf_thr", type=float, default=0.2, help="Fallback confidence threshold for weak hand keypoints when high-confidence points are sparse")
    parser.add_argument("--hand_kpt_hi_min_keypoints", type=int, default=6, help="Use high-confidence hand bbox when at least this many keypoints exceed hand_kpt_conf_thr")
    parser.add_argument("--hand_min_keypoints", type=int, default=4, help="Minimum confident hand keypoints before HaMeR crop is valid")
    parser.add_argument("--hamer_bbox_rescale", type=float, default=2.0, help="HaMeR hand bbox rescale factor")
    parser.add_argument(
        "--hamer_bbox_rescale_candidates",
        type=str,
        default="",
        help="Comma-separated HaMeR bbox rescale candidates; the one with lowest 2D keypoint error is selected per hand frame.",
    )
    parser.add_argument("--hamer_candidate_switch_penalty", type=float, default=8.0, help="Pixel-cost penalty for switching HaMeR bbox-scale candidate between adjacent frames")
    parser.add_argument("--hand_bbox_min_size", type=float, default=0.0, help="Minimum hand bbox side length in input pixels; 0 disables")
    parser.add_argument("--hand_bbox_smoothing", type=float, default=0.0, help="Temporal smoothing for valid hand bboxes. 0 disables; 0.6 keeps 60% previous bbox")
    parser.add_argument("--hand_bbox_max_jump", type=float, default=0.0, help="Maximum bbox center jump in input pixels before clipping. 0 disables")
    parser.add_argument("--hand_bbox_overlap_iou", type=float, default=0.55, help="IoU threshold for rejecting left/right hand bbox collisions. 0 disables")
    parser.add_argument("--hand_bbox_collision_score_ratio", type=float, default=1.35, help="Confidence ratio needed to suppress the weaker hand when left/right bboxes overlap")
    parser.add_argument("--hamer_batch_size", type=int, default=2, help="HaMeR/WiLoR dataloader batch size")
    parser.add_argument("--hamer_refine_steps", type=int, default=0, help="Post-HaMeR MANO 2D fitting steps; 0 disables")
    parser.add_argument("--hamer_refine_lr", type=float, default=0.03, help="Learning rate for post-HaMeR MANO 2D fitting")
    parser.add_argument("--hamer_refine_conf_thr", type=float, default=0.45, help="Only hand keypoints above this confidence drive 2D fitting")
    parser.add_argument("--hamer_refine_min_keypoints", type=int, default=6, help="Minimum confident keypoints needed to refine one hand crop")
    parser.add_argument("--hamer_refine_pose_prior", type=float, default=0.02, help="Prior weight keeping refined finger pose near HaMeR")
    parser.add_argument("--hamer_refine_global_prior", type=float, default=0.01, help="Prior weight keeping refined wrist orientation near HaMeR")
    args = parser.parse_args()
    
    # Input
    video_path = Path(args.video)
    fps = get_video_fps(video_path) if not args.recreate_video else 30
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        video_name = video_path.stem if args.video_name is None else args.video_name
        overrides = [
            f"video_name={hydra_quote(video_name)}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"+batch_size={args.batch_size}",
            f"+fps={round(fps)}",
            f"+export_pt={args.export_pt}",
            f"+skip_render={args.skip_render}",
            f"+low_memory={args.low_memory}",
            f"+force_hand_preprocess={args.force_hand_preprocess}",
            f"+hand_preprocess_only={args.hand_preprocess_only}",
            f"+render_only={args.render_only}",
            f"+hand_backend={args.hand_backend}",
            f"+wilor_root={hydra_quote(args.wilor_root)}",
            f"+wilor_checkpoint={hydra_quote(args.wilor_checkpoint)}",
            f"+wilor_config={hydra_quote(args.wilor_config)}",
            f"+wilor_fast={args.wilor_fast}",
            f"+hand4wholepp_root={hydra_quote(args.hand4wholepp_root)}",
            f"+hand4wholepp_snapshot={hydra_quote(args.hand4wholepp_snapshot)}",
            f"+hand4wholepp_python={hydra_quote(args.hand4wholepp_python)}",
            f"+hand4wholepp_batch_size={args.hand4wholepp_batch_size}",
            f"+hand4wholepp_yolo_model={hydra_quote(args.hand4wholepp_yolo_model)}",
            f"+hand4wholepp_joint_source={hydra_quote(args.hand4wholepp_joint_source)}",
            f"+hand4wholepp_crop_tracking={hydra_quote(args.hand4wholepp_crop_tracking)}",
            f"+hand4wholepp_crop_tracking_max_gap={args.hand4wholepp_crop_tracking_max_gap}",
            f"+hand4wholepp_crop_tracking_max_prediction_gap={args.hand4wholepp_crop_tracking_max_prediction_gap}",
            f"+hand4wholepp_crop_tracking_direct_observation_quality={args.hand4wholepp_crop_tracking_direct_observation_quality}",
            f"+vitpose_img_ds={args.vitpose_img_ds}",
            f"+hand_kpt_conf_thr={args.hand_kpt_conf_thr}",
            f"+hand_kpt_low_conf_thr={args.hand_kpt_low_conf_thr}",
            f"+hand_kpt_hi_min_keypoints={args.hand_kpt_hi_min_keypoints}",
            f"+hand_min_keypoints={args.hand_min_keypoints}",
            f"+hamer_bbox_rescale={args.hamer_bbox_rescale}",
            f"+hamer_bbox_rescale_candidates={hydra_quote(args.hamer_bbox_rescale_candidates)}",
            f"+hamer_candidate_switch_penalty={args.hamer_candidate_switch_penalty}",
            f"+hand_bbox_min_size={args.hand_bbox_min_size}",
            f"+hand_bbox_smoothing={args.hand_bbox_smoothing}",
            f"+hand_bbox_max_jump={args.hand_bbox_max_jump}",
            f"+hand_bbox_overlap_iou={args.hand_bbox_overlap_iou}",
            f"+hand_bbox_collision_score_ratio={args.hand_bbox_collision_score_ratio}",
            f"+hamer_batch_size={args.hamer_batch_size}",
            f"+hamer_refine_steps={args.hamer_refine_steps}",
            f"+hamer_refine_lr={args.hamer_refine_lr}",
            f"+hamer_refine_conf_thr={args.hamer_refine_conf_thr}",
            f"+hamer_refine_min_keypoints={args.hamer_refine_min_keypoints}",
            f"+hamer_refine_pose_prior={args.hamer_refine_pose_prior}",
            f"+hamer_refine_global_prior={args.hamer_refine_global_prior}",
        ]

        # Allow to change output root
        if args.output_root is not None:
            overrides.append(f"output_root={hydra_quote(args.output_root)}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)
        if args.mano_params_override:
            with open_dict(cfg):
                cfg.paths.mano_params = str(Path(args.mano_params_override).expanduser().resolve())

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        # create a soft link
        if args.recreate_video:
            reader = get_video_reader(video_path)
            writer = get_writer(cfg.video_path, fps=30, crf=CRF)
            for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
                writer.write_frame(img)
            writer.close()
            reader.close()
        else:
            try:
                os.symlink(os.path.abspath(video_path), os.path.abspath(cfg.video_path))
            except FileExistsError:
                os.remove(os.path.abspath(cfg.video_path))
                os.symlink(os.path.abspath(video_path), os.path.abspath(cfg.video_path))
    
    valid_video_path = os.path.abspath(cfg.video_path).replace('0_input_video.mp4', 'valid_video.mp4')
    try:
        shutil.copy2(os.path.abspath(cfg.video_path), valid_video_path)
    except FileExistsError:
        os.remove(valid_video_path)
        shutil.copy2(os.path.abspath(cfg.video_path), valid_video_path)
    
    return cfg


def _cfg_path(cfg, key, default_path):
    value = str(cfg.get(key, "") or "").strip()
    if value:
        return Path(value).expanduser().resolve()
    return Path(default_path).expanduser().resolve()


def _load_hamer_backend():
    if _load_hamer is None:
        raise ImportError(
            "HaMeR is not importable. Run scripts/setup_hamer_assets.sh and ensure "
            "third-party/hamer is installed or on PYTHONPATH."
        ) from _HAMER_IMPORT_ERROR
    return _load_hamer(DEFAULT_CHECKPOINT_HAMER)


def _load_wilor_backend(cfg):
    gvhmr_root = Path(__file__).resolve().parents[2]
    wilor_root = _cfg_path(cfg, "wilor_root", gvhmr_root / "third-party" / "WiLoR")
    checkpoint = _cfg_path(cfg, "wilor_checkpoint", wilor_root / "pretrained_models" / "wilor_final.ckpt")
    config = _cfg_path(cfg, "wilor_config", wilor_root / "pretrained_models" / "model_config.yaml")

    missing = [
        path
        for path in (
            wilor_root / "wilor" / "models" / "__init__.py",
            checkpoint,
            config,
            wilor_root / "mano_data" / "MANO_RIGHT.pkl",
            wilor_root / "mano_data" / "mano_mean_params.npz",
        )
        if not path.exists()
    ]
    if missing:
        pretty = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "WiLoR backend is not ready. Missing:\n  "
            f"{pretty}\nRun: bash scripts/setup_wilor_assets.sh"
        )

    if str(wilor_root) not in sys.path:
        sys.path.insert(0, str(wilor_root))

    cwd = os.getcwd()
    try:
        os.chdir(wilor_root)
        from wilor.models import load_wilor

        model, model_cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(config))
    finally:
        os.chdir(cwd)

    if bool(cfg.get("wilor_fast", False)):
        torch.set_float32_matmul_precision("high")
        if hasattr(model, "backbone"):
            model.backbone.skip_blocks = True
    return model, model_cfg


def _run_hand4wholepp_backend(cfg, video_path, bbox_path, vitpose_path, output_path):
    gvhmr_root = Path(__file__).resolve().parents[2]
    h4w_root = _cfg_path(
        cfg,
        "hand4wholepp_root",
        gvhmr_root / "third-party" / "Hand4Whole-plus-plus_RELEASE",
    )
    snapshot = _cfg_path(cfg, "hand4wholepp_snapshot", h4w_root / "demo" / "snapshot_6.pth")
    python = str(cfg.get("hand4wholepp_python", "") or "").strip() or sys.executable
    runner = Path(__file__).resolve().with_name("run_hand4wholepp_video.py")

    cmd = [
        python,
        str(runner),
        "--video",
        str(video_path),
        "--output",
        str(output_path),
        "--root",
        str(h4w_root),
        "--snapshot",
        str(snapshot),
        "--bbox_pt",
        str(bbox_path),
        "--vitpose_wholebody",
        str(vitpose_path),
        "--batch_size",
        str(int(cfg.get("hand4wholepp_batch_size", 1))),
        "--yolo_model",
        str(cfg.get("hand4wholepp_yolo_model", "yolo11n.pt")),
        "--joint_source",
        str(cfg.get("hand4wholepp_joint_source", "direct_mano")),
        "--hand_crop_tracking",
        str(cfg.get("hand4wholepp_crop_tracking", "off")),
        "--hand_crop_tracking_max_gap",
        str(int(cfg.get("hand4wholepp_crop_tracking_max_gap", 8))),
        "--hand_crop_tracking_max_prediction_gap",
        str(int(cfg.get("hand4wholepp_crop_tracking_max_prediction_gap", 2))),
        "--hand_crop_tracking_direct_observation_quality",
        str(float(cfg.get("hand4wholepp_crop_tracking_direct_observation_quality", 0.75))),
    ]
    Log.info("[Preprocess] running Hand4Whole++ backend")
    Log.info("[Preprocess] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(gvhmr_root), check=True)


def load_hand_backend(cfg):
    backend = str(cfg.get("hand_backend", "hamer")).lower()
    if backend == "hamer":
        model, model_cfg = _load_hamer_backend()
    elif backend == "wilor":
        model, model_cfg = _load_wilor_backend(cfg)
    else:
        raise ValueError(f"Unsupported hand_backend={backend!r}")
    return backend, model, model_cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose
    batch_size = cfg.batch_size
    cropped_imgs = None

    if cfg.get("force_hand_preprocess", False):
        for path in (paths.vitpose_wholebody, paths.vitpose, paths.mano_params):
            path = Path(path)
            if path.exists():
                path.unlink()
                Log.info(f"[Preprocess] removed cached hand input: {path}")

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy, bbx_conf = tracker.get_all_tracks(video_path, frame_thres=0.5)  # (P, L, 4), (P, L) discard short tracks
        bbx_xys = get_bbx_xys_from_xyxy_batch(bbx_xyxy, base_enlarge=1.2).float()  # (P, L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy.detach().cpu(), "bbx_xys": bbx_xys.detach().cpu(), "bbx_conf": bbx_conf.detach().cpu()}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx, weights_only=True)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx, weights_only=True)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch_multiperson(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay, fps=cfg.fps)
    person_num = bbx_xys.shape[0]
    print(f"person_num: {person_num}")
    def chunk_first_axis(tensor, person_num):
        # (frame_num*person_num, ..) -> (frame_num, person_num, ..)
        return tensor.reshape(-1, person_num, *tensor.shape[1:])
    
    # Get VitPose-wholebody (0-16: 17 body keypoints, 17-22: 6 foot keypoints, 23-90: 68 face keypoints, 91-132: 42 hand keypoints)
    if not Path(paths.vitpose_wholebody).exists():
        vitpose_extractor = VitPoseWholebodyExtractor(batch_size=batch_size)
        vitpose_wholebody, cropped_imgs = vitpose_extractor.extract_multiperson(
            video_path,
            bbx_xys,
            img_ds=float(cfg.get("vitpose_img_ds", 0.5)),
        )  # (P, F, 133, 3)
        torch.save(vitpose_wholebody.detach().cpu(), paths.vitpose_wholebody)
        del vitpose_extractor
    else:
        vitpose_wholebody = torch.load(paths.vitpose_wholebody, weights_only=True)
        Log.info(f"[Preprocess] vitpose-wholebody from {paths.vitpose_wholebody}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco133_skeleton_batch(video, vitpose_wholebody.transpose(0, 1), 0.5, 2, 4)
        save_video(video_overlay, paths.vitpose_wholebody_video_overlay, fps=cfg.fps)
    
    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose = vitpose_wholebody[:, :, :17, :]  # (P, F, 17, 3)
        torch.save(vitpose.detach().cpu(), paths.vitpose)
    else:
        vitpose = torch.load(paths.vitpose, weights_only=True)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose.transpose(0, 1), 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay, fps=cfg.fps)
    
    # Get mano params
    if not Path(paths.mano_params).exists():
        hand_backend = str(cfg.get("hand_backend", "hamer")).lower()
        if hand_backend == "hand4wholepp":
            # Hand4Whole++ is launched as a child process and loads several
            # multi-gigabyte checkpoints.  Release the finished ViTPose model
            # and CUDA caching allocator before that child starts so both
            # networks are not resident at the same time.
            cropped_imgs = None
            release_torch_memory(
                "[Preprocess] released ViTPose cache before Hand4Whole++"
            )
            _run_hand4wholepp_backend(
                cfg,
                video_path,
                paths.bbx,
                paths.vitpose_wholebody,
                paths.mano_params,
            )
            release_torch_memory("[Preprocess] released Hand4Whole++ runner memory")
        else:
            hand_backend, hand_model, model_cfg_hand = load_hand_backend(cfg)
            Log.info(f"[Preprocess] hand backend: {hand_backend}")
            hand_model = hand_model.cuda()
            hand_model.eval()
            # create hand dataset
            frames = read_frames(video_path)
            hamer_rescales = parse_float_candidates(
                cfg.get("hamer_bbox_rescale_candidates", ""),
                cfg.get("hamer_bbox_rescale", 2.0),
            )
            candidate_mano_params = []
            for hamer_rescale in hamer_rescales:
                hamer_dataloader = load_images(
                    frames,
                    vitpose_wholebody,
                    model_cfg_hand,
                    hand_kpt_conf_thr=float(cfg.get("hand_kpt_conf_thr", 0.5)),
                    hand_kpt_low_conf_thr=float(cfg.get("hand_kpt_low_conf_thr", 0.2)),
                    hand_kpt_hi_min_keypoints=int(cfg.get("hand_kpt_hi_min_keypoints", 6)),
                    hand_min_keypoints=int(cfg.get("hand_min_keypoints", 4)),
                    hamer_bbox_rescale=float(hamer_rescale),
                    hand_bbox_min_size=float(cfg.get("hand_bbox_min_size", 0.0)),
                    hand_bbox_smoothing=float(cfg.get("hand_bbox_smoothing", 0.0)),
                    hand_bbox_max_jump=float(cfg.get("hand_bbox_max_jump", 0.0)),
                    hand_bbox_overlap_iou=float(cfg.get("hand_bbox_overlap_iou", 0.55)),
                    hand_bbox_collision_score_ratio=float(cfg.get("hand_bbox_collision_score_ratio", 1.35)),
                    hamer_batch_size=int(cfg.get("hamer_batch_size", 64)),
                )
                all_mano_params = {'left_hand_global_orient': [], 'left_hand_pose': [], 'left_hand_joints_3d': [], 'left_hand_valid': [],
                                   'right_hand_global_orient': [], 'right_hand_pose': [], 'right_hand_joints_3d': [], 'right_hand_valid': [],
                                   'left_hand_reproj_error': [], 'right_hand_reproj_error': [],
                                   'left_hand_bbox_xyxy': [], 'right_hand_bbox_xyxy': []}
                for batch in tqdm(hamer_dataloader, desc=f"{hand_backend} x{hamer_rescale:g}"):
                    batch = recursive_to(batch, target="cuda")
                    mano_poses = predict_mano(
                        batch,
                        hand_model,
                        refine_steps=int(cfg.get("hamer_refine_steps", 0)),
                        refine_lr=float(cfg.get("hamer_refine_lr", 0.03)),
                        refine_conf_thr=float(cfg.get("hamer_refine_conf_thr", 0.45)),
                        refine_min_keypoints=int(cfg.get("hamer_refine_min_keypoints", 6)),
                        refine_pose_prior=float(cfg.get("hamer_refine_pose_prior", 0.02)),
                        refine_global_prior=float(cfg.get("hamer_refine_global_prior", 0.01)),
                    )
                    for k, v in mano_poses.items():
                        all_mano_params[k].append(chunk_first_axis(v, person_num))
                for k, v in all_mano_params.items():
                    all_mano_params[k] = torch.cat(v, dim=0).transpose(0, 1).detach().cpu()
                candidate_mano_params.append(all_mano_params)

            all_mano_params = select_best_hamer_candidate(
                candidate_mano_params,
                hamer_rescales,
                switch_penalty=float(cfg.get("hamer_candidate_switch_penalty", 8.0)),
            )
            all_mano_params = temporal_hold_invalid_mano_frames(all_mano_params)
            torch.save(all_mano_params, paths.mano_params)
            del hand_model, model_cfg_hand, frames, candidate_mano_params, all_mano_params
            release_torch_memory(f"[Preprocess] released {hand_backend} memory")
    else:
        Log.info(f"[Preprocess] mano_params from {paths.mano_params}")

    if cfg.get("hand_preprocess_only", False):
        Log.info("[Preprocess] hand_preprocess_only=True; stop after hand MANO params")
        return

    if cfg.get("low_memory", False):
        cropped_imgs = None
        del vitpose_wholebody, vitpose
        release_torch_memory("[Preprocess] low_memory cleanup before HMR2 features")
    
    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor(batch_size=batch_size)
        inputs = cropped_imgs if cropped_imgs is not None else video_path
        vit_features = extractor.extract_video_features_multiperson(inputs, bbx_xys)  # (P, F, 1024)
        torch.save(vit_features.detach().cpu(), paths.vit_features)
        del extractor, inputs, vit_features
        release_torch_memory("[Preprocess] released HMR2 feature extractor memory")
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get DPVO results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
            bar = tqdm(total=length, desc="DPVO")
            while True:
                ret = slam.track()
                if ret:
                    bar.update()
                else:
                    break
            slam_results = slam.process()  # (L, 7), numpy
            torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx, weights_only=True)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose, weights_only=True),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features, weights_only=True),
    }
    return data


def fetch_smpl_params(all_person_dict, person_idx):
    params = {}
    for k, v in all_person_dict.items():
        value = v[person_idx]
        if k in {"global_orient", "body_pose"} and value.shape[-2:] == (3, 3):
            value = matrix_to_axis_angle(value)
        params[k] = value
    return params


def load_mano_if_available(cfg):
    """Load mano_params.pt if it exists and contains hand data."""
    mano_path = Path(cfg.paths.mano_params)
    if not mano_path.is_file():
        Log.info(f"[Render] mano_params not found at {mano_path}, rendering body-only")
        return None
    mano = torch.load(mano_path, map_location="cpu", weights_only=False)
    if "left_hand_pose" not in mano and "right_hand_pose" not in mano:
        Log.info("[Render] mano_params has no hand_pose keys, rendering body-only")
        return None
    Log.info(f"[Render] Loaded mano_params from {mano_path}")
    return mano


def fetch_smpl_params_with_hands(all_person_dict, person_idx, mano_params, frame_count, smplx_model=None):
    """Merge hand4wholepp/HaMeR hand data into SMPL-X params for rendering.

    Returns a dict of per-frame SMPL-X parameters ready for the SMPL-X model.
    """
    from pytorch3d.transforms import axis_angle_to_matrix

    params = {}
    for k, v in all_person_dict.items():
        value = v[person_idx]  # (F, ...)
        if k in {"global_orient", "body_pose"} and value.shape[-2:] == (3, 3):
            value = matrix_to_axis_angle(value)
        params[k] = value

    # Add hand poses from mano_params (convert rotation matrices to PCA or flat axis-angle)
    for side, hand_key, wrist_idx in [
        ("left", "left_hand_pose", 19),
        ("right", "right_hand_pose", 20),
    ]:
        if hand_key in mano_params:
            hp = mano_params[hand_key][person_idx]  # (F, 15, 3, 3)
            hp_aa = matrix_to_axis_angle(hp.reshape(-1, 3, 3)).reshape(
                hp.shape[0], 45
            )  # (F, 45) flat axis-angle
            hp_aa = hp_aa[:frame_count].float()

            # Convert to PCA coefficients if the SMPL-X model uses PCA
            if smplx_model is not None and getattr(smplx_model.bm, 'use_pca', False):
                components = getattr(smplx_model.bm, f'{side}_hand_components', None)
                if components is not None:
                    # components: (num_pca_comps, 45); pc = aa @ components^T
                    hp_aa = torch.matmul(
                        hp_aa.to(components.device), components.T
                    ).cpu()  # (F, num_pca_comps)

            params[f"{side}_hand_pose"] = hp_aa

    # Replace wrist joints in body_pose with hand4wholepp wrist orientation
    body_pose = params.get("body_pose")  # (F, 21, 3) axis-angle
    global_orient = params.get("global_orient")  # (F, 3) axis-angle
    if body_pose is not None and global_orient is not None:
        go_mat = axis_angle_to_matrix(global_orient.reshape(-1, 3)).reshape(
            global_orient.shape[0], 3, 3  # (F, 3, 3) — simpler, no extra dim
        )
        bp_mat = axis_angle_to_matrix(body_pose.reshape(-1, 3)).reshape(
            body_pose.shape[0], 21, 3, 3
        )

        left_chain = np.array([3, 6, 9, 13, 16, 18]) - 1
        right_chain = np.array([3, 6, 9, 14, 17, 19]) - 1
        left_global_key = (
            "left_hand_global_orient_filtered"
            if "left_hand_global_orient_filtered" in mano_params
            else "left_hand_global_orient"
        )
        right_global_key = (
            "right_hand_global_orient_filtered"
            if "right_hand_global_orient_filtered" in mano_params
            else "right_hand_global_orient"
        )

        for chain, wrist_idx, global_key, valid_key in [
            (left_chain, 19, left_global_key, "left_hand_valid"),
            (right_chain, 20, right_global_key, "right_hand_valid"),
        ]:
            if global_key not in mano_params:
                continue
            # hand_global: (F, 3, 3) or (F, 1, 3, 3) depending on backend
            hand_global = mano_params[global_key][person_idx].float()
            if hand_global.ndim == 4 and hand_global.shape[1] == 1:
                hand_global = hand_global[:, 0]  # (F, 1, 3, 3) -> (F, 3, 3)
            # Truncate/pad to frame_count
            hg = hand_global[:frame_count]
            if hg.shape[0] < frame_count:
                pad = torch.eye(3).unsqueeze(0).repeat(frame_count - hg.shape[0], 1, 1)
                hg = torch.cat([hg, pad], dim=0)

            valid = mano_params.get(valid_key)
            if valid is not None:
                valid = valid[person_idx].bool()[:frame_count]

            # Compute body-derived wrist pose per frame and replace
            for f in range(frame_count):
                if valid is not None and not valid[f]:
                    continue
                # Build cumulative rotation from global_orient through chain
                wrist_rot = go_mat[f : f + 1]  # (1, 3, 3)
                for idx in chain:
                    wrist_rot = wrist_rot @ bp_mat[f : f + 1, idx]  # (1, 3, 3) @ (1, 3, 3)
                # Replace body_pose wrist with hand4wholepp orientation
                inv_wrist = torch.inverse(wrist_rot.float())  # (1, 3, 3)
                bp_mat[f : f + 1, wrist_idx] = inv_wrist @ hg[f : f + 1]  # (1, 3, 3) @ (1, 3, 3)

        # Convert back to axis-angle
        params["body_pose"] = matrix_to_axis_angle(
            bp_mat.reshape(-1, 3, 3)
        ).reshape(bp_mat.shape[0], 21, 3)

    return params


def create_merged_faces(faces_smpl, person_num, vert_offset):
    """
    Create merged faces for rendering multiple persons.
    
    Args:
        faces_smpl (numpy.ndarray): The original faces of the SMPL model, shape (face_num, 3).
        person_num (int): The number of persons to be rendered.
        vert_offset (int): The vertex offset for the current person.
    Returns:
        numpy.ndarray: The merged faces, shape (face_num * person_num, 3).
    """
    merged_faces = []
    for i in range(person_num):
        merged_faces.append(faces_smpl + i * vert_offset)
    return np.concatenate(merged_faces, axis=0)
    

def retarget_transl(global_transl, incam_transl_start, xz_only=False):
    """
    Retarget the start point of the global translation to the start point of the incam translation.
    
    Args:
        global_transl (torch.Tensor): The global translation, shape (P, F, 3).
        incam_transl_start (torch.Tensor): The incam translation start point, shape (P, 3).
    Returns:
        torch.Tensor: The retargeted global translation, shape (P, F, 3).
    """
    if xz_only:
        index = [0, 2]
    else:
        index = [0, 1, 2]
    retargeted_transl = global_transl.clone()
    retargeted_transl[:, 0, index] = incam_transl_start[:, index]
    retargeted_transl[:, 1:, index] = global_transl[:, 1:, index] + (incam_transl_start[:, None, index] - global_transl[:, 0:1, index])
    return retargeted_transl


def render_incam(cfg, retarget=False):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results, weights_only=True)
    # Hand4Whole++ supplies a full 15x3 MANO pose.  The SuperMotion default
    # uses a 12-D hand PCA space, which silently projects that pose before
    # rendering and visibly distorts fingers.  Keep the body model unchanged,
    # but render the hand in its native 45-D axis-angle representation.
    smplx = make_smplx("supermotion", use_pca=False).cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=True).cuda()   # (6890, 10475)
    faces_smpl = make_smplx("smpl").faces   # (face_num, 3)
    mano_params = load_mano_if_available(cfg)

    # smpl
    if retarget:
        global_transl = pred["smpl_params_global"]["transl"]
        incam_transl_start = pred["smpl_params_incam"]["transl"][:, 0]
        pred["smpl_params_incam"]["transl"] = retarget_transl(global_transl, incam_transl_start, xz_only=False)

    person_num = pred["smpl_params_incam"]["transl"].shape[0]
    frame_num = pred["smpl_params_incam"]["transl"].shape[1]
    merged_verts = []
    for person_idx in range(person_num):
        if mano_params is not None:
            smplx_params = fetch_smpl_params_with_hands(
                pred["smpl_params_incam"], person_idx, mano_params, frame_num, smplx
            )
        else:
            smplx_params = fetch_smpl_params(pred["smpl_params_incam"], person_idx)
        smplx_out = smplx(**to_cuda(smplx_params))
        pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])  # (F, 6890, 3)
        merged_verts.append(pred_c_verts)
    pred_c_verts = torch.stack(merged_verts, dim=1).reshape(frame_num, -1, 3)  # (F, P, 6890, 3) -> (F, P*6890, 3)
    faces_smpl = create_merged_faces(faces_smpl, person_num, smplx2smpl.shape[0])
    
    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx, weights_only=True)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts  # (F, V, 3)
    writer = get_writer(incam_video_path, fps=cfg.fps, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg, retarget=False):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    # See render_incam(): preserving the full MANO hand pose is required for
    # the global render to match the direct-MANO track as well.
    smplx = make_smplx("supermotion", use_pca=False).cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=True).cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=True).cuda()
    mano_params = load_mano_if_available(cfg)

    # smpl
    global_transl = pred["smpl_params_global"]["transl"]
    incam_transl_start = pred["smpl_params_incam"]["transl"][:, 0]
    if retarget:
        pred["smpl_params_global"]["transl"] = retarget_transl(global_transl, incam_transl_start, xz_only=False)

    person_num = pred["smpl_params_global"]["transl"].shape[0]
    frame_num = pred["smpl_params_global"]["transl"].shape[1]
    merged_verts = []
    for person_idx in range(person_num):
        if mano_params is not None:
            smplx_params = fetch_smpl_params_with_hands(
                pred["smpl_params_global"], person_idx, mano_params, frame_num, smplx
            )
        else:
            smplx_params = fetch_smpl_params(pred["smpl_params_global"], person_idx)
        smplx_out = smplx(**to_cuda(smplx_params))
        pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
        merged_verts.append(pred_ay_verts)
    pred_ay_verts = torch.stack(merged_verts, dim=1)  # (F, P, V, 3)
    
    # position
    all_offset = []
    for person_idx in range(person_num):
        verts = pred_ay_verts[:, person_idx].clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        all_offset.append(offset)
    offset = torch.mean(torch.stack(all_offset, dim=0), dim=0)
    
    verts_glob = pred_ay_verts - offset
        
    joints_glob = einsum(J_regressor, verts_glob[:, 0], "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob[:, 0].cpu(),
        beta=3.0,
        cam_height_degree=25,
        target_center_height=1.0,
    )
    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 18)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob[:, 0])
    renderer.set_ground(scale * 4.0, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=cfg.fps, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[i], color[None].repeat(person_num, 1), cameras, global_lights)
        writer.write_frame(img)
    writer.close()


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def parse_float_candidates(text, fallback):
    text = str(text or "").strip()
    if not text:
        return [float(fallback)]
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    return values or [float(fallback)]


def gather_candidate(values, best_idx):
    stacked = torch.stack(values, dim=0)
    index = best_idx.long().unsqueeze(0)
    for _ in range(stacked.ndim - index.ndim):
        index = index.unsqueeze(-1)
    index = index.expand((1, *stacked.shape[1:]))
    return torch.gather(stacked, dim=0, index=index).squeeze(0)


def viterbi_candidate_indices(errors, switch_penalty=8.0):
    """Choose temporally stable candidate ids for errors shaped (K, P, T)."""
    if errors.ndim != 3:
        return torch.argmin(errors, dim=0)
    if errors.shape[0] <= 1 or switch_penalty <= 0:
        return torch.argmin(errors, dim=0)

    errors_np = errors.detach().float().cpu().numpy()
    k_count, person_count, frame_count = errors_np.shape
    paths = np.zeros((person_count, frame_count), dtype=np.int64)
    transition = np.full((k_count, k_count), float(switch_penalty), dtype=np.float32)
    np.fill_diagonal(transition, 0.0)

    for person_idx in range(person_count):
        unary = errors_np[:, person_idx]
        finite = np.isfinite(unary)
        if not np.any(finite):
            continue
        safe_unary = np.where(finite, unary, 1e6).astype(np.float32)
        dp = np.zeros((k_count, frame_count), dtype=np.float32)
        back = np.zeros((k_count, frame_count), dtype=np.int64)
        dp[:, 0] = safe_unary[:, 0]
        for frame_idx in range(1, frame_count):
            prev = dp[:, frame_idx - 1][:, None] + transition
            back[:, frame_idx] = np.argmin(prev, axis=0)
            dp[:, frame_idx] = safe_unary[:, frame_idx] + prev[back[:, frame_idx], np.arange(k_count)]
        best = int(np.argmin(dp[:, -1]))
        for frame_idx in range(frame_count - 1, -1, -1):
            paths[person_idx, frame_idx] = best
            best = int(back[best, frame_idx])
    return torch.from_numpy(paths).to(errors.device)


def select_best_hamer_candidate(candidates, candidate_scales, switch_penalty=8.0):
    if len(candidates) == 1:
        out = dict(candidates[0])
        out["hamer_bbox_rescale_candidates"] = torch.tensor(candidate_scales, dtype=torch.float32)
        return out

    selected = {}
    for side in ("left", "right"):
        error_key = f"{side}_hand_reproj_error"
        errors = torch.stack([c[error_key].float() for c in candidates], dim=0)
        best_idx = viterbi_candidate_indices(errors, switch_penalty=float(switch_penalty))
        for key in (
            f"{side}_hand_global_orient",
            f"{side}_hand_pose",
            f"{side}_hand_joints_3d",
            f"{side}_hand_valid",
            error_key,
            f"{side}_hand_bbox_xyxy",
        ):
            selected[key] = gather_candidate([c[key] for c in candidates], best_idx)
        selected[f"{side}_hand_candidate_idx"] = best_idx.cpu()

    selected["hamer_bbox_rescale_candidates"] = torch.tensor(candidate_scales, dtype=torch.float32)
    return selected


def temporal_hold_invalid_mano_frames(mano_params):
    """Replace weak-evidence HaMeR frames with the nearest previous reliable hand state."""
    out = dict(mano_params)
    for side in ("left", "right"):
        valid_key = f"{side}_hand_valid"
        if valid_key not in out:
            continue
        valid = out[valid_key].bool()
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)

        for key in (
            f"{side}_hand_global_orient",
            f"{side}_hand_pose",
            f"{side}_hand_joints_3d",
            f"{side}_hand_bbox_xyxy",
        ):
            if key not in out:
                continue
            values = out[key].clone()
            squeezed_person = False
            if values.shape[0] != valid.shape[0] and valid.shape[0] == 1:
                values = values.unsqueeze(0)
                squeezed_person = True

            for person_idx in range(valid.shape[0]):
                valid_idx = torch.nonzero(valid[person_idx], as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue
                first = int(valid_idx[0])
                if first > 0:
                    values[person_idx, :first] = values[person_idx, first]
                last = first
                for frame_idx in range(first + 1, valid.shape[1]):
                    if bool(valid[person_idx, frame_idx]):
                        last = frame_idx
                    else:
                        values[person_idx, frame_idx] = values[person_idx, last]

            out[key] = values.squeeze(0) if squeezed_person else values

        out[f"{side}_hand_temporal_hold_mask"] = (~valid).cpu()
    return out


def expand_bbox_xyxy(bbox, min_size=0.0):
    bbox = np.asarray(bbox, dtype=np.float32)
    if min_size <= 0.0:
        return bbox
    cx = float((bbox[0] + bbox[2]) * 0.5)
    cy = float((bbox[1] + bbox[3]) * 0.5)
    width = max(float(bbox[2] - bbox[0]), float(min_size))
    height = max(float(bbox[3] - bbox[1]), float(min_size))
    half = max(width, height) * 0.5
    return np.asarray([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def bbox_from_hand_keypoints(
    keyp,
    conf_thr=0.5,
    low_conf_thr=0.2,
    hi_min_keypoints=6,
    min_keypoints=4,
    min_size=0.0,
):
    if isinstance(keyp, torch.Tensor):
        keyp = keyp.detach().cpu().numpy()
    keyp = np.asarray(keyp, dtype=np.float32)
    finite_xy = np.isfinite(keyp[:, :2]).all(axis=1)
    conf = np.where(finite_xy & np.isfinite(keyp[:, 2]), keyp[:, 2], -np.inf)
    valid_hi = conf > float(conf_thr)
    valid_lo = conf > float(low_conf_thr)
    selected = valid_hi if int(valid_hi.sum()) >= int(hi_min_keypoints) else valid_lo
    if int(selected.sum()) >= int(min_keypoints):
        pts = keyp[selected, :2]
        bbox = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        return expand_bbox_xyxy(bbox, min_size), True
    return None, False


def hand_keypoint_evidence_score(keyp, conf_thr=0.5, low_conf_thr=0.2, hi_min_keypoints=6):
    if isinstance(keyp, torch.Tensor):
        keyp = keyp.detach().cpu().numpy()
    keyp = np.asarray(keyp, dtype=np.float32)
    finite_xy = np.isfinite(keyp[:, :2]).all(axis=1)
    conf = np.where(finite_xy & np.isfinite(keyp[:, 2]), keyp[:, 2], 0.0)
    valid_hi = conf > float(conf_thr)
    valid_lo = conf > float(low_conf_thr)
    selected = valid_hi if int(valid_hi.sum()) >= int(hi_min_keypoints) else valid_lo
    count = int(selected.sum())
    if count <= 0:
        return 0.0
    # Mean confidence alone cannot distinguish a dense hand from one or two
    # noisy points, so include a mild count term.
    return float(np.mean(conf[selected]) * np.sqrt(count))


def bbox_from_wrist_elbow(vitposes, side, min_size=96.0, conf_thr=0.2):
    if isinstance(vitposes, torch.Tensor):
        vitposes = vitposes.detach().cpu().numpy()
    vitposes = np.asarray(vitposes, dtype=np.float32)
    if vitposes.shape[0] < 11:
        return None

    if side == "left":
        wrist_idx, elbow_idx, shoulder_idx = 9, 7, 5
    else:
        wrist_idx, elbow_idx, shoulder_idx = 10, 8, 6

    wrist = vitposes[wrist_idx]
    if not np.isfinite(wrist[:2]).all() or float(wrist[2]) <= float(conf_thr):
        return None

    size = max(float(min_size), 1.0)
    for idx, scale in ((elbow_idx, 1.6), (shoulder_idx, 0.8)):
        joint = vitposes[idx]
        if np.isfinite(joint[:2]).all() and float(joint[2]) > float(conf_thr):
            dist = float(np.linalg.norm(wrist[:2] - joint[:2]))
            if np.isfinite(dist) and dist > 1.0:
                size = max(size, dist * scale)
                break

    cx, cy = float(wrist[0]), float(wrist[1])
    half = size * 0.5
    return np.asarray([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)


def body_anchor_from_keypoints(vitposes, conf_thr=0.2):
    if isinstance(vitposes, torch.Tensor):
        vitposes = vitposes.detach().cpu().numpy()
    vitposes = np.asarray(vitposes, dtype=np.float32)
    torso_indices = [5, 6, 11, 12]
    pts = []
    for idx in torso_indices:
        if idx < vitposes.shape[0] and np.isfinite(vitposes[idx, :2]).all() and float(vitposes[idx, 2]) > float(conf_thr):
            pts.append(vitposes[idx, :2])
    if len(pts) >= 2:
        return np.mean(np.asarray(pts, dtype=np.float32), axis=0)

    body = vitposes[:17] if vitposes.shape[0] >= 17 else vitposes
    valid = np.isfinite(body[:, :2]).all(axis=1) & (body[:, 2] > float(conf_thr))
    if int(valid.sum()) >= 2:
        return np.mean(body[valid, :2], axis=0)
    return None


def transport_bbox_by_anchor(prev_bbox, prev_anchor, curr_anchor):
    if prev_bbox is None or prev_anchor is None or curr_anchor is None:
        return None
    prev_bbox = np.asarray(prev_bbox, dtype=np.float32)
    prev_anchor = np.asarray(prev_anchor, dtype=np.float32)
    curr_anchor = np.asarray(curr_anchor, dtype=np.float32)
    if not (np.isfinite(prev_bbox).all() and np.isfinite(prev_anchor).all() and np.isfinite(curr_anchor).all()):
        return None
    delta = curr_anchor - prev_anchor
    return prev_bbox + np.asarray([delta[0], delta[1], delta[0], delta[1]], dtype=np.float32)


def bbox_iou_xyxy(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


def bbox_center_and_side(bbox):
    bbox = np.asarray(bbox, dtype=np.float32)
    center = (bbox[:2] + bbox[2:]) * 0.5
    side = max(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]), 1.0)
    return center, side


def wrist_distance_ratio_to_bbox(vitposes, side, bbox, conf_thr=0.2):
    if bbox is None:
        return np.inf
    if isinstance(vitposes, torch.Tensor):
        vitposes = vitposes.detach().cpu().numpy()
    vitposes = np.asarray(vitposes, dtype=np.float32)
    wrist_idx = 9 if side == "left" else 10
    wrist = vitposes[wrist_idx]
    if not np.isfinite(wrist[:2]).all() or float(wrist[2]) <= float(conf_thr):
        return np.inf
    center, side_len = bbox_center_and_side(bbox)
    return float(np.linalg.norm(center - wrist[:2]) / side_len)


def hand_bbox_support_score(bbox, keyp, vitposes, side, conf_thr=0.5, low_conf_thr=0.2, hi_min_keypoints=6):
    evidence = hand_keypoint_evidence_score(
        keyp,
        conf_thr=conf_thr,
        low_conf_thr=low_conf_thr,
        hi_min_keypoints=hi_min_keypoints,
    )
    wrist_ratio = wrist_distance_ratio_to_bbox(vitposes, side, bbox)
    if np.isfinite(wrist_ratio):
        body_support = float(np.exp(-1.5 * wrist_ratio))
    else:
        body_support = 0.5
    return evidence * max(body_support, 0.05)


def choose_non_colliding_fallback(
    side,
    current_bbox,
    winner_bbox,
    prev_bboxes,
    prev_bbox_anchors,
    person_idx,
    vitposes,
    body_anchor,
    hand_bbox_min_size,
    overlap_iou,
):
    transported = transport_bbox_by_anchor(
        prev_bboxes.get((person_idx, side)),
        prev_bbox_anchors.get((person_idx, side)),
        body_anchor,
    )
    if transported is not None and bbox_iou_xyxy(transported, winner_bbox) < float(overlap_iou):
        return transported
    wrist_bbox = bbox_from_wrist_elbow(vitposes, side, min_size=hand_bbox_min_size)
    if wrist_bbox is not None and bbox_iou_xyxy(wrist_bbox, winner_bbox) < float(overlap_iou):
        return wrist_bbox
    prev = prev_bboxes.get((person_idx, side))
    if prev is not None and bbox_iou_xyxy(prev, winner_bbox) < float(overlap_iou):
        return np.asarray(prev, dtype=np.float32).copy()
    if transported is not None:
        return transported
    if prev is not None:
        return np.asarray(prev, dtype=np.float32).copy()
    return current_bbox


def resolve_hand_bbox_collision(
    left_bbox,
    left_valid,
    right_bbox,
    right_valid,
    left_keyp,
    right_keyp,
    vitposes,
    prev_bboxes,
    prev_bbox_anchors,
    person_idx,
    body_anchor,
    hand_kpt_conf_thr=0.5,
    hand_kpt_low_conf_thr=0.2,
    hand_kpt_hi_min_keypoints=6,
    hand_bbox_min_size=0.0,
    overlap_iou=0.55,
    score_ratio=1.35,
):
    if overlap_iou <= 0.0 or bbox_iou_xyxy(left_bbox, right_bbox) < float(overlap_iou):
        return left_bbox, left_valid, right_bbox, right_valid

    left_score = hand_bbox_support_score(
        left_bbox,
        left_keyp,
        vitposes,
        "left",
        conf_thr=hand_kpt_conf_thr,
        low_conf_thr=hand_kpt_low_conf_thr,
        hi_min_keypoints=hand_kpt_hi_min_keypoints,
    )
    right_score = hand_bbox_support_score(
        right_bbox,
        right_keyp,
        vitposes,
        "right",
        conf_thr=hand_kpt_conf_thr,
        low_conf_thr=hand_kpt_low_conf_thr,
        hi_min_keypoints=hand_kpt_hi_min_keypoints,
    )

    left_own = wrist_distance_ratio_to_bbox(vitposes, "left", left_bbox)
    left_other = wrist_distance_ratio_to_bbox(vitposes, "right", left_bbox)
    right_own = wrist_distance_ratio_to_bbox(vitposes, "right", right_bbox)
    right_other = wrist_distance_ratio_to_bbox(vitposes, "left", right_bbox)
    left_crossed = np.isfinite(left_own) and np.isfinite(left_other) and left_other + 0.25 < left_own
    right_crossed = np.isfinite(right_own) and np.isfinite(right_other) and right_other + 0.25 < right_own

    loser = None
    if left_valid and not right_valid:
        loser = "right"
    elif right_valid and not left_valid:
        loser = "left"
    elif left_crossed and not right_crossed:
        loser = "left"
    elif right_crossed and not left_crossed:
        loser = "right"
    elif left_score * float(score_ratio) < right_score:
        loser = "left"
    elif right_score * float(score_ratio) < left_score:
        loser = "right"

    if loser is None:
        left_center, left_side = bbox_center_and_side(left_bbox)
        right_center, right_side = bbox_center_and_side(right_bbox)
        center_dist = float(np.linalg.norm(left_center - right_center))
        same_crop = center_dist < 0.25 * min(left_side, right_side)
        if same_crop and bbox_iou_xyxy(left_bbox, right_bbox) >= max(float(overlap_iou), 0.75):
            if np.isfinite(left_own) and np.isfinite(right_own):
                if left_own > right_own + 0.15:
                    loser = "left"
                elif right_own > left_own + 0.15:
                    loser = "right"
            if loser is None:
                loser = "left" if left_score < right_score else "right"

    if loser == "left":
        left_bbox = choose_non_colliding_fallback(
            "left",
            left_bbox,
            right_bbox,
            prev_bboxes,
            prev_bbox_anchors,
            person_idx,
            vitposes,
            body_anchor,
            hand_bbox_min_size,
            overlap_iou,
        )
        left_valid = False
    elif loser == "right":
        right_bbox = choose_non_colliding_fallback(
            "right",
            right_bbox,
            left_bbox,
            prev_bboxes,
            prev_bbox_anchors,
            person_idx,
            vitposes,
            body_anchor,
            hand_bbox_min_size,
            overlap_iou,
        )
        right_valid = False

    return left_bbox, left_valid, right_bbox, right_valid


def stabilize_bbox_with_previous(bbox, prev_bbox, smoothing=0.0, max_center_jump=0.0):
    if bbox is None or prev_bbox is None:
        return bbox
    bbox = np.asarray(bbox, dtype=np.float32).copy()
    prev_bbox = np.asarray(prev_bbox, dtype=np.float32)

    prev_center = (prev_bbox[:2] + prev_bbox[2:]) * 0.5
    center = (bbox[:2] + bbox[2:]) * 0.5
    size = bbox[2:] - bbox[:2]
    delta = center - prev_center
    jump = float(np.linalg.norm(delta))
    if max_center_jump and max_center_jump > 0.0 and jump > float(max_center_jump):
        center = prev_center + delta * (float(max_center_jump) / max(jump, 1e-6))
        bbox[:2] = center - size * 0.5
        bbox[2:] = center + size * 0.5

    alpha = float(np.clip(smoothing, 0.0, 0.95))
    if alpha > 0.0:
        bbox = alpha * prev_bbox + (1.0 - alpha) * bbox
    return bbox.astype(np.float32)


def load_images(
    cv2_images,
    wholebody_kpts,
    model_cfg,
    hand_kpt_conf_thr=0.5,
    hand_kpt_low_conf_thr=0.2,
    hand_kpt_hi_min_keypoints=6,
    hand_min_keypoints=4,
    hamer_bbox_rescale=2.0,
    hand_bbox_min_size=0.0,
    hand_bbox_smoothing=0.0,
    hand_bbox_max_jump=0.0,
    hand_bbox_overlap_iou=0.55,
    hand_bbox_collision_score_ratio=1.35,
    hamer_batch_size=64,
):
    hand_datasets = []
    prev_bboxes = {}
    prev_bbox_anchors = {}

    for idx, img_cv2 in enumerate(cv2_images):
        vitposes_out = wholebody_kpts[:, idx]  # (P, F, 133, 3) -> (P, 133, 3)
        person_num = vitposes_out.shape[0]
        bboxes, is_right, is_valid, hand_keypoints = [], [], [], []
        for person_idx in range(person_num):
            vitposes = vitposes_out[person_idx]
            body_anchor = body_anchor_from_keypoints(vitposes)
            left_hand_keyp = vitposes[-42:-21]
            right_hand_keyp = vitposes[-21:]

            left_bbox, left_valid = bbox_from_hand_keypoints(
                left_hand_keyp,
                conf_thr=hand_kpt_conf_thr,
                low_conf_thr=hand_kpt_low_conf_thr,
                hi_min_keypoints=hand_kpt_hi_min_keypoints,
                min_keypoints=hand_min_keypoints,
                min_size=hand_bbox_min_size,
            )
            if left_bbox is None:
                left_bbox = transport_bbox_by_anchor(
                    prev_bboxes.get((person_idx, "left")),
                    prev_bbox_anchors.get((person_idx, "left")),
                    body_anchor,
                )
                left_valid = False
            if left_bbox is None:
                left_bbox = bbox_from_wrist_elbow(vitposes, "left", min_size=hand_bbox_min_size)
                left_valid = False
            if left_bbox is None:
                left_bbox = prev_bboxes.get((person_idx, "left"))
                left_valid = False
            if left_bbox is None:
                height, width = img_cv2.shape[:2]
                side = max(float(hand_bbox_min_size), min(height, width) * 0.2)
                cx, cy = width * 0.5, height * 0.5
                left_bbox = np.asarray([cx - side, cy - side, cx + side, cy + side], dtype=np.float32)
                left_valid = False
            if left_valid:
                left_bbox = stabilize_bbox_with_previous(
                    left_bbox,
                    prev_bboxes.get((person_idx, "left")),
                    smoothing=hand_bbox_smoothing,
                    max_center_jump=hand_bbox_max_jump,
                )
            
            right_bbox, right_valid = bbox_from_hand_keypoints(
                right_hand_keyp,
                conf_thr=hand_kpt_conf_thr,
                low_conf_thr=hand_kpt_low_conf_thr,
                hi_min_keypoints=hand_kpt_hi_min_keypoints,
                min_keypoints=hand_min_keypoints,
                min_size=hand_bbox_min_size,
            )
            if right_bbox is None:
                right_bbox = transport_bbox_by_anchor(
                    prev_bboxes.get((person_idx, "right")),
                    prev_bbox_anchors.get((person_idx, "right")),
                    body_anchor,
                )
                right_valid = False
            if right_bbox is None:
                right_bbox = bbox_from_wrist_elbow(vitposes, "right", min_size=hand_bbox_min_size)
                right_valid = False
            if right_bbox is None:
                right_bbox = prev_bboxes.get((person_idx, "right"))
                right_valid = False
            if right_bbox is None:
                height, width = img_cv2.shape[:2]
                side = max(float(hand_bbox_min_size), min(height, width) * 0.2)
                cx, cy = width * 0.5, height * 0.5
                right_bbox = np.asarray([cx - side, cy - side, cx + side, cy + side], dtype=np.float32)
                right_valid = False
            if right_valid:
                right_bbox = stabilize_bbox_with_previous(
                    right_bbox,
                    prev_bboxes.get((person_idx, "right")),
                    smoothing=hand_bbox_smoothing,
                    max_center_jump=hand_bbox_max_jump,
                )

            left_bbox, left_valid, right_bbox, right_valid = resolve_hand_bbox_collision(
                left_bbox,
                left_valid,
                right_bbox,
                right_valid,
                left_hand_keyp,
                right_hand_keyp,
                vitposes,
                prev_bboxes,
                prev_bbox_anchors,
                person_idx,
                body_anchor,
                hand_kpt_conf_thr=hand_kpt_conf_thr,
                hand_kpt_low_conf_thr=hand_kpt_low_conf_thr,
                hand_kpt_hi_min_keypoints=hand_kpt_hi_min_keypoints,
                hand_bbox_min_size=hand_bbox_min_size,
                overlap_iou=hand_bbox_overlap_iou,
                score_ratio=hand_bbox_collision_score_ratio,
            )

            if left_valid:
                prev_bboxes[(person_idx, "left")] = np.asarray(left_bbox, dtype=np.float32).copy()
                if body_anchor is not None:
                    prev_bbox_anchors[(person_idx, "left")] = np.asarray(body_anchor, dtype=np.float32).copy()
            bboxes.append(left_bbox)
            is_right.append(0)
            is_valid.append(left_valid)
            hand_keypoints.append(left_hand_keyp.detach().cpu().numpy())

            if right_valid:
                prev_bboxes[(person_idx, "right")] = np.asarray(right_bbox, dtype=np.float32).copy()
                if body_anchor is not None:
                    prev_bbox_anchors[(person_idx, "right")] = np.asarray(body_anchor, dtype=np.float32).copy()
            bboxes.append(right_bbox)
            is_right.append(1)
            is_valid.append(right_valid)
            hand_keypoints.append(right_hand_keyp.detach().cpu().numpy())

        if len(bboxes) == 0:
            bboxes, right, valid = np.empty((0, 4)), np.empty(0), np.empty(0)
            hand_keypoints = np.empty((0, 21, 3), dtype=np.float32)
        else:
            bboxes, right, valid = np.array(bboxes), np.array(is_right), np.array(is_valid)
            hand_keypoints = np.asarray(hand_keypoints, dtype=np.float32)
            
        hand_dataset = ViTDetDataset(
            model_cfg,
            img_cv2,
            bboxes,
            right,
            valid,
            keypoints_2d=hand_keypoints,
            rescale_factor=hamer_bbox_rescale,
        )
        hand_datasets.append(hand_dataset)

    concatenated_hand_dataset = ConcatDataset(hand_datasets)
    hand_dataloader = DataLoader(concatenated_hand_dataset, batch_size=hamer_batch_size, shuffle=False, num_workers=0)
    return hand_dataloader


def _project_mano_keypoints_full_image(model, batch, global_aa, hand_aa, betas, pred_cam):
    """Project MANO joints into the original image frame using the hand crop camera."""
    try:
        from hamer.utils.geometry import perspective_projection
    except ImportError:
        from wilor.utils.geometry import perspective_projection

    batch_size = global_aa.shape[0]
    device = global_aa.device
    dtype = global_aa.dtype

    global_orient = axis_angle_to_matrix(global_aa.reshape(-1, 3)).reshape(batch_size, 1, 3, 3)
    hand_pose = axis_angle_to_matrix(hand_aa.reshape(-1, 3)).reshape(batch_size, 15, 3, 3)
    mano_output = model.mano(
        global_orient=global_orient.float(),
        hand_pose=hand_pose.float(),
        betas=betas.float(),
        pose2rot=False,
    )

    focal_length = model.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
    pred_cam_t = torch.stack(
        [
            pred_cam[:, 1],
            pred_cam[:, 2],
            2 * focal_length[:, 0] / (model.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
        ],
        dim=-1,
    )

    pred_keypoints_crop = perspective_projection(
        mano_output.joints,
        translation=pred_cam_t,
        focal_length=focal_length / model.cfg.MODEL.IMAGE_SIZE,
    ).reshape(batch_size, -1, 2)

    pred_keypoints_full = pred_keypoints_crop.clone()
    right = batch["right"].float()
    pred_keypoints_full[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_full[:, :, 0]
    pred_keypoints_full = (
        pred_keypoints_full * batch["box_size"].float()[:, None, None]
        + batch["box_center"].float()[:, None, :]
    )
    return pred_keypoints_crop, pred_keypoints_full


def refine_mano_to_2d(
    batch,
    model,
    out,
    steps=0,
    lr=0.03,
    conf_thr=0.45,
    min_keypoints=6,
    pose_prior=0.02,
    global_prior=0.01,
):
    """Lightweight test-time MANO fitting against ViTPose wholebody hand keypoints."""
    if steps <= 0:
        return out

    mano_params = out["pred_mano_params"]
    init_global = mano_params["global_orient"].detach().reshape(-1, 1, 3, 3)
    init_hand = mano_params["hand_pose"].detach().reshape(-1, 15, 3, 3)
    init_betas = mano_params["betas"].detach().reshape(init_global.shape[0], -1)
    init_cam = out["pred_cam"].detach()
    batch_size = init_global.shape[0]

    target = batch["hand_keypoints_2d"].float()
    conf = torch.clamp(target[:, :, 2], min=0.0)
    valid_crop = batch["valid"].float() > 0.5
    confident = conf > float(conf_thr)
    refine_mask = valid_crop & (confident.sum(dim=1) >= int(min_keypoints))
    if int(refine_mask.sum().item()) == 0:
        return out

    weights = conf * confident.float() * refine_mask.float()[:, None]
    box_size = torch.clamp(batch["box_size"].float(), min=1.0)

    with torch.enable_grad():
        global_aa0 = matrix_to_axis_angle(init_global.reshape(-1, 3, 3)).reshape(batch_size, 1, 3)
        hand_aa0 = matrix_to_axis_angle(init_hand.reshape(-1, 3, 3)).reshape(batch_size, 15, 3)
        global_aa = global_aa0.detach().clone().requires_grad_(True)
        hand_aa = hand_aa0.detach().clone().requires_grad_(True)

        optimizer = torch.optim.Adam([global_aa, hand_aa], lr=float(lr))
        denom = torch.clamp(weights.sum(), min=1.0)
        refine_weight = refine_mask.float()[:, None, None]

        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            _, pred_full = _project_mano_keypoints_full_image(
                model, batch, global_aa, hand_aa, init_betas, init_cam
            )
            diff = (pred_full - target[:, :, :2]) / box_size[:, None, None]
            reproj_loss = (torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-8) * weights).sum() / denom
            hand_prior = (((hand_aa - hand_aa0) ** 2) * refine_weight).mean()
            global_prior_loss = (((global_aa - global_aa0) ** 2) * refine_weight).mean()
            loss = reproj_loss + float(pose_prior) * hand_prior + float(global_prior) * global_prior_loss
            loss.backward()
            optimizer.step()

        refined_global = axis_angle_to_matrix(global_aa.detach().reshape(-1, 3)).reshape(batch_size, 1, 3, 3)
        refined_hand = axis_angle_to_matrix(hand_aa.detach().reshape(-1, 3)).reshape(batch_size, 15, 3, 3)
        mask_global = refine_mask[:, None, None, None]
        mask_hand = refine_mask[:, None, None, None]
        mano_params["global_orient"] = torch.where(mask_global, refined_global, init_global)
        mano_params["hand_pose"] = torch.where(mask_hand, refined_hand, init_hand)

        pred_crop, _ = _project_mano_keypoints_full_image(
            model,
            batch,
            matrix_to_axis_angle(mano_params["global_orient"].reshape(-1, 3, 3)).reshape(batch_size, 1, 3),
            matrix_to_axis_angle(mano_params["hand_pose"].reshape(-1, 3, 3)).reshape(batch_size, 15, 3),
            init_betas,
            init_cam,
        )
        out["pred_keypoints_2d"] = torch.where(
            refine_mask[:, None, None], pred_crop.detach(), out["pred_keypoints_2d"].detach()
        )
        out["pred_mano_params"] = mano_params

    return out


def predict_mano(
    batch,
    model,
    refine_steps=0,
    refine_lr=0.03,
    refine_conf_thr=0.45,
    refine_min_keypoints=6,
    refine_pose_prior=0.02,
    refine_global_prior=0.01,
):
    batch_size = batch['img'].shape[0]
    with torch.no_grad():
        out = model(batch)
    out = refine_mano_to_2d(
        batch,
        model,
        out,
        steps=refine_steps,
        lr=refine_lr,
        conf_thr=refine_conf_thr,
        min_keypoints=refine_min_keypoints,
        pose_prior=refine_pose_prior,
        global_prior=refine_global_prior,
    )
    flip_idx = batch['right'] == 0
    invalid_idx = batch['valid'] == 0   # weak visual evidence; downstream filters decide how to use it
    assert flip_idx.sum() == batch_size // 2, f"flip_idx: {flip_idx.sum()}, batch_size: {batch_size}"
    
    mano_params = out['pred_mano_params']
    global_orient = mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
    hand_pose = mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
    betas = mano_params['betas'].reshape(batch_size, -1)
    # Keep HaMeR's estimate even for weak visual evidence.  The valid mask and
    # reprojection error below tell downstream temporal filters how much to
    # trust the frame; zeroing here causes visible open-hand snaps.
    mano_output = model.mano(
        global_orient=global_orient.float(),
        hand_pose=hand_pose.float(),
        betas=betas.float(),
        pose2rot=False,
    )
    joints_3d = mano_output.joints.detach().clone()
    pred_keypoints_2d = out['pred_keypoints_2d'].detach().clone()
    right = batch['right'].float()
    pred_keypoints_2d[:, :, 0] = (2 * right[:, None] - 1) * pred_keypoints_2d[:, :, 0]
    pred_keypoints_2d = pred_keypoints_2d * batch['box_size'][:, None, None] + batch['box_center'][:, None, :]
    target_keypoints_2d = batch['hand_keypoints_2d'].to(pred_keypoints_2d.device).float()
    weights = torch.clamp(target_keypoints_2d[:, :, 2], min=0.0)
    weights = weights * batch['valid'].float()[:, None]
    reproj_dist = torch.linalg.norm(pred_keypoints_2d - target_keypoints_2d[:, :, :2], dim=-1)
    reproj_error = (reproj_dist * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1e-6)
    reproj_error[invalid_idx] = 1e6
    # Flip the hand pose
    reflection_matrix = torch.tensor([
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, 1]
    ], dtype=torch.float32).cuda()
    global_orient[flip_idx] = torch.einsum('bkij,ij->bkij', global_orient[flip_idx], reflection_matrix)
    hand_pose[flip_idx] = torch.einsum('bkij,ij->bkij', hand_pose[flip_idx], reflection_matrix)
    joints_3d[flip_idx, :, 0] *= -1.0
    mano_poses = {
        'left_hand_global_orient': global_orient[flip_idx],
        'left_hand_pose': hand_pose[flip_idx],
        'left_hand_joints_3d': joints_3d[flip_idx],
        'right_hand_global_orient': global_orient[~flip_idx],
        'right_hand_pose': hand_pose[~flip_idx],
        'right_hand_joints_3d': joints_3d[~flip_idx],
        'left_hand_valid': batch['valid'][flip_idx],
        'right_hand_valid': batch['valid'][~flip_idx],
        'left_hand_reproj_error': reproj_error[flip_idx],
        'right_hand_reproj_error': reproj_error[~flip_idx],
        'left_hand_bbox_xyxy': batch['bbox'][flip_idx].float(),
        'right_hand_bbox_xyxy': batch['bbox'][~flip_idx].float(),
    }
    return mano_poses


def convert_final_results(pred):
    for smpl_type in ["smpl_params_global", "smpl_params_incam"]:
        smpl_params = pred[smpl_type]
        person_num, frame_num = smpl_params["transl"].shape[0], smpl_params["transl"].shape[1]
        global_orient = smpl_params["global_orient"]
        body_pose = smpl_params["body_pose"]
        smpl_params["body_pose"] = axis_angle_to_mat3x3(body_pose.reshape(-1, 3)).reshape(person_num, frame_num, 21, 3, 3)
        smpl_params["global_orient"] = axis_angle_to_mat3x3(global_orient.reshape(-1, 3)).reshape(person_num, frame_num, 1, 3, 3)
        pred[smpl_type] = smpl_params
    pred['width'] = pred['K_fullimg'][:, 0, 2] * 2
    pred['height'] = pred['K_fullimg'][:, 1, 2] * 2
    pred['focal_length'] = pred['K_fullimg'][:, 0, 0]
    return pred


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

    if cfg.get("render_only", False):
        if not Path(paths.hmr4d_results).is_file():
            raise FileNotFoundError(f"render_only requires existing HMR4D results: {paths.hmr4d_results}")
        if cfg.skip_render:
            raise ValueError("render_only cannot be combined with --skip_render")
        render_incam(cfg)
        render_global(cfg, retarget=True)
        if not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
        raise SystemExit(0)

    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    if cfg.get("hand_preprocess_only", False):
        raise SystemExit(0)
    release_torch_memory("[Main] cleanup before loading GVHMR predictor")
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict_multiperson(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        pred['fps'] = cfg.fps
        data_time = data["length"] / cfg.fps
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(convert_final_results(pred), paths.hmr4d_results)

    # ===== Render ===== #
    if cfg.skip_render:
        # delete 0_input_video.mp4
        subprocess.run(["rm", cfg.video_path])
    else:
        retarget = True
        render_incam(cfg)
        render_global(cfg, retarget=retarget)
        if not Path(paths.incam_global_horiz_video).exists():
            Log.info("[Merge Videos]")
            merge_videos_horizontal([paths.incam_video, paths.global_video], paths.incam_global_horiz_video)
    if cfg.export_pt:
        verts_pt_path = os.path.join(cfg.output_dir, "smpl_verts.pt")
        subprocess.run(["python", "-m", "tools.processor.export_pt_verts", "--input", paths.hmr4d_results, "--output", verts_pt_path, "--mano_params", paths.mano_params])
    
"""
CUDA_VISIBLE_DEVICES=2, python -m tools.processor.generate_smplxs --video=docs/example_video/vertical_dance.mp4 --output_root outputs/demo_mp -s
CUDA_VISIBLE_DEVICES=6, python -m tools.processor.generate_smplxs --video=docs/example_video/two_persons.mp4 --output_root outputs/demo_mp_hands --skip_render --export_pt
CUDA_VISIBLE_DEVICES=7, python -m tools.processor.generate_smplxs --video=docs/example_video/tiktok_frame.mp4 --output_root outputs/demo_single_frame --skip_render -s
CUDA_VISIBLE_DEVICES=3, python -m tools.processor.generate_smplxs --video=/mnt/data/jing/Video_Generation/video_data_repos/video_preprocessor/WHAM/examples/dance2.mp4 --output_root outputs/demo_mp
"""



import os.path as osp
from pathlib import Path
from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_amp_task as humanoid_amp_task
from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_real import MotionLibReal
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from easydict import EasyDict

from phc.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from collections import defaultdict

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import open3d as o3d
from datetime import datetime
import imageio
from collections import deque
from tqdm import tqdm
import copy
import cv2
import time


def _resolve_fix_height_mode(cfg):
    name = cfg.get("phc_fix_height", cfg["env"].get("phc_fix_height", "full_fix"))
    if isinstance(name, FixHeightMode):
        return name
    name = str(name).strip().lower()
    aliases = {
        "none": "no_fix",
        "no": "no_fix",
        "off": "no_fix",
        "no_fix": "no_fix",
        "full": "full_fix",
        "full_fix": "full_fix",
        "ankle": "ankle_fix",
        "ankle_fix": "ankle_fix",
    }
    key = aliases.get(name)
    if key is None:
        raise ValueError(f"Unsupported phc_fix_height={name!r}; expected full_fix, ankle_fix, or no_fix")
    return getattr(FixHeightMode, key)


class HumanoidIm(humanoid_amp_task.HumanoidAMPTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._full_body_reward = cfg["env"].get("full_body_reward", True)
        self._fut_tracks = cfg["env"].get("fut_tracks", False)
        self._fut_tracks_dropout = cfg["env"].get("fut_tracks_dropout", False)
        self.seq_motions = cfg["env"].get("seq_motions", False)
        if self._fut_tracks:
            self._num_traj_samples = cfg["env"]["numTrajSamples"]
        else:
            self._num_traj_samples = 1
        self._min_motion_len = cfg["env"].get("min_length", -1)
        self._traj_sample_timestep = 1 / cfg["env"].get("trajSampleTimestepInv", 30)

        self.load_humanoid_configs(cfg)
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.headless = cfg["headless"]
        #self.start_idx = 0

        self.reward_specs = cfg["env"].get("reward_specs", {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1, "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1})

        self._num_joints = len(self._body_names)
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self._track_bodies = cfg["env"].get("trackBodies", self._full_track_bodies)
        self._track_bodies_id = self._build_key_body_ids_tensor(self._track_bodies)
        self._reset_bodies = cfg["env"].get("reset_bodies", self._track_bodies)
        self._reset_bodies_id = self._build_key_body_ids_tensor(self._reset_bodies)
        
        self._full_track_bodies_id = self._build_key_body_ids_tensor(self._full_track_bodies)
        self._eval_track_bodies_id = self._build_key_body_ids_tensor(self._eval_bodies)
        self._motion_start_times_offset = torch.zeros(self.num_envs).to(self.device)
        self._cycle_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        
        if "extend_config" in cfg.robot:
            extend_names, extend_pos, extend_rot = [], [], []
            for extend_config in cfg.robot.extend_config:
                extend_names.append(extend_config["parent_name"])
                extend_pos.append(extend_config["pos"])
            
            self.extend_body_parent_ids = self._build_key_body_ids_tensor(extend_names)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.num_extend_bodies = len(extend_names)

        spacing = 5
        side_lenght = torch.ceil(torch.sqrt(torch.tensor(self.num_envs)))
        pos_x, pos_y = torch.meshgrid(torch.arange(side_lenght) * spacing, torch.arange(side_lenght) * spacing)
        self.start_pos_x, self.start_pos_y = pos_x.flatten(), pos_y.flatten()
        self._global_offset = torch.zeros([self.num_envs, 3]).to(self.device)
        # self._global_offset[:, 0], self._global_offset[:, 1] = self.start_pos_x[:self.num_envs], self.start_pos_y[:self.num_envs]

        self.offset_range = 0.8

        ## ZL Hack Devs
        #################### Devs ####################
        self._point_goal = torch.zeros(self.num_envs, device=self.device)
        self.random_occlu_idx = torch.zeros((self.num_envs, len(self._track_bodies)), device=self.device, dtype=torch.bool)
        self.random_occlu_count = torch.zeros((self.num_envs, len(self._track_bodies)), device=self.device).long()
        #################### Devs ####################

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        
        if self.humanoid_type in ['h1', 'g1', ]:
            self.actions = torch.zeros(self.num_envs, self._dof_obs_size).to(self.device) #### Keeping taps on previous actions
            
        # Overriding
        self.reward_raw = torch.zeros((self.num_envs, 5 if self.power_reward else 4)).to(self.device)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)

        if (not self.headless or flags.server_mode):
            self._build_marker_state_tensors()
        
        self.ref_body_pos = torch.zeros_like(self._rigid_body_pos)
        self.ref_body_vel = torch.zeros_like(self._rigid_body_vel)
        self.ref_body_rot = torch.zeros_like(self._rigid_body_rot)
        self.ref_body_pos_subset = torch.zeros_like(self._rigid_body_pos[:, self._track_bodies_id])
        self.ref_dof_pos = torch.zeros_like(self._dof_pos)

        self.viewer_o3d = flags.render_o3d
        self.vis_ref = True
        self.vis_contact = False
        self.control_i = 0
        self.auto_record_motion = cfg.get("auto_record_motion", False)
        self.wait_for_o3d_close_after_record = cfg.get("wait_for_o3d_close_after_record", False)
        self._auto_record_started = False
        self._auto_record_done = False
        self._auto_record_num_steps = None
        self._auto_record_frames = 0
        self._pending_o3d_auto_record_exit = False
        self._o3d_interactive_mode = False  # Non-blocking: keep O3D open while sim runs
        # Auto-rotate camera during O3D recording
        self.o3d_auto_rotate = cfg.get("o3d_auto_rotate", False)
        self.o3d_rotation_speed = float(cfg.get("o3d_rotation_speed", 0.0))
        self.o3d_num_rotations = float(cfg.get("o3d_num_rotations", 2.5))
        self.o3d_camera_mode = str(cfg.get("o3d_camera_mode", "level")).strip().lower()
        if self.o3d_camera_mode not in {"level", "topdown"}:
            print(f"[WARN] Unsupported o3d_camera_mode={self.o3d_camera_mode!r}; falling back to level")
            self.o3d_camera_mode = "level"
        default_elevation = 78.0 if self.o3d_camera_mode == "topdown" else 26.0
        default_zoom = 0.82 if self.o3d_camera_mode == "topdown" else 0.82
        self.o3d_camera_elevation_deg = float(cfg.get("o3d_camera_elevation_deg", default_elevation))
        self.o3d_camera_zoom = float(cfg.get("o3d_camera_zoom", default_zoom))
        default_lookat_ratio = 0.50 if self.o3d_camera_mode == "topdown" else 0.38
        self.o3d_camera_lookat_height_ratio = float(cfg.get("o3d_camera_lookat_height_ratio", default_lookat_ratio))
        self.o3d_camera_lookat_z_offset = float(cfg.get("o3d_camera_lookat_z_offset", 0.0))
        self.o3d_camera_follow = bool(cfg.get("o3d_camera_follow", True))
        self.o3d_camera_follow_alpha = float(cfg.get("o3d_camera_follow_alpha", cfg.get("o3d_camera_follow_smoothing", 0.10)))
        self.o3d_floor_style = str(cfg.get("o3d_floor_style", "solid")).strip().lower()
        self._o3d_rotation_angle = float(cfg.get("o3d_camera_yaw_deg", 0.0))
        self._o3d_last_center = None
        self._o3d_target_center = None
        self._sampled_motion_ids = torch.arange(self.num_envs).to(self.device)
        self.create_o3d_viewer()
        return

    def export_repaired_motion_for_current_sample(self):
        if len(self.state_record.get("progress", [])) == 0:
            print("[WARN] state_record is empty, skip export for current sample")
            return None

        curr_motion_keys = self._motion_lib.curr_motion_keys
        if isinstance(curr_motion_keys, (str, bytes, np.str_)):
            motion_key = str(curr_motion_keys)
        elif getattr(curr_motion_keys, "shape", None) == ():
            motion_key = str(curr_motion_keys.item())
        else:
            motion_key = str(curr_motion_keys[0])
        motion_root = self.cfg.get("repair_motion_root", "")
        if not motion_root:
            raise ValueError("repair_motion_root must be set when seq_export_states=True")

        motion_key_path = Path(motion_key)
        motion_root_path = Path(motion_root).resolve()
        try:
            rel_path = motion_key_path.resolve().relative_to(motion_root_path)
        except Exception:
            rel_path = Path(motion_key_path.name)

        state_root = Path(self.cfg.get("repair_state_root", "output/states"))
        state_file = state_root / rel_path.parent / f"{motion_key_path.stem}.pkl"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Dumping sequential state into {state_file}")
        self._write_states_to_file(state_file.as_posix())

        repaired_root = self.cfg.get("repair_output_root", "")
        reference_root = self.cfg.get("repair_reference_root", "")
        if repaired_root and reference_root:
            from scripts.data_process.batch_repair_zitai import export_longest_state_to_npz
            repaired_file = Path(repaired_root) / rel_path.parent / f"{motion_key_path.stem}_repaired.npz"
            reference_file = Path(reference_root) / rel_path.parent / f"{motion_key_path.stem}.npz"
            repaired_file.parent.mkdir(parents=True, exist_ok=True)
            export_longest_state_to_npz(
                state_file=state_file,
                output_file=repaired_file,
                reference_npz=reference_file,
                gravity_axis=self.cfg.get("repair_output_gravity_axis", "neg_z"),
            )
            print(f"[OK] {reference_file} -> {repaired_file}")
            return repaired_file.as_posix()

        return state_file.as_posix()
    
    
    def pause_func(self, action):
        self.paused = not self.paused
        
    def next_func(self, action):
        self.resample_motions()
    
    def reset_func(self, action):
        self.reset()

    def record_func(self, action):
        self.recording = not self.recording
        self.recording_state_change_o3d = True
        self.recording_state_change_o3d_img = True
        self.recording_state_change = True # only intialize from o3d. 
        
        
    def hide_ref(self, action):
        flags.show_traj = not flags.show_traj

    def _create_o3d_floor(self):
        ground_size = float(self.cfg.get("o3d_floor_size", 5.0))
        floor_z = float(self.cfg.get("o3d_floor_z", -0.03))
        style = self.o3d_floor_style

        if style == "solid":
            height = float(self.cfg.get("o3d_floor_thickness", 0.004))
            floor = o3d.geometry.TriangleMesh.create_box(width=ground_size, height=ground_size, depth=height)
            floor.translate(np.array([-ground_size / 2, -ground_size / 2, floor_z - height]))
            floor.compute_vertex_normals()
            floor.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.74, 0.74, 0.74]]).repeat(8, axis=0))
            return floor

        step = max(float(self.cfg.get("o3d_floor_grid_step", 0.25)), 1e-3)
        half = ground_size / 2.0
        count = int(np.floor(ground_size / step))
        values = np.linspace(-half, half, count + 1)
        points, lines, colors = [], [], []

        def add_line(a, b, color):
            idx = len(points)
            points.extend([a, b])
            lines.append([idx, idx + 1])
            colors.append(color)

        for value in values:
            major = abs(value) < step * 0.5
            color = [0.30, 0.30, 0.30] if major else [0.62, 0.62, 0.62]
            add_line([-half, value, floor_z], [half, value, floor_z], color)
            add_line([value, -half, floor_z], [value, half, floor_z], color)

        floor = o3d.geometry.LineSet()
        floor.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        floor.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        floor.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        return floor
    
    def create_o3d_viewer(self):
        ################################################ ZL Hack: o3d viewers. ################################################
        if self.viewer_o3d :
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
            self.o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
            self.o3d_vis.create_window()
            
            floor = self._create_o3d_floor()
            
            
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
                
                if self.humanoid_type == "smpl":
                    self.mujoco_2_smpl = [self._body_names_orig.index(q) for q in SMPL_BONE_ORDER_NAMES if q in self._body_names_orig]
                elif self.humanoid_type in ["smplh", "smplx"]:
                    self.mujoco_2_smpl = [self._body_names_orig.index(q) for q in SMPLH_BONE_ORDER_NAMES if q in self._body_names_orig]

                with torch.no_grad():
                    verts, joints = self._motion_lib.mesh_parsers[0].get_joints_verts(pose = torch.zeros(1, len(self._body_names_orig) * 3))
                    np_triangles = self._motion_lib.mesh_parsers[0].faces
                if self._has_upright_start:
                    self.pre_rot = sRot.from_quat([0.5, 0.5, 0.5, 0.5])
                else:
                    self.pre_rot = sRot.identity()
                self.mesh_parser = copy.deepcopy(self._motion_lib.mesh_parsers[0])
                self.mesh_parser = self.mesh_parser.cuda()
            
            self.sim_mesh = o3d.geometry.TriangleMesh()
            self.sim_mesh.vertices = o3d.utility.Vector3dVector(verts.numpy()[0])
            self.sim_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
            self.sim_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0, 0.5, 0.5]]).repeat(verts.shape[1], axis=0))
            if self.vis_ref:
                self.ref_mesh = o3d.geometry.TriangleMesh()
                self.ref_mesh.vertices = o3d.utility.Vector3dVector(verts.numpy()[0])
                self.ref_mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
                self.ref_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[0.5, 0., 0.]]).repeat(verts.shape[1], axis=0))
                self.o3d_vis.add_geometry(self.ref_mesh)

            self.o3d_vis.add_geometry(floor)
            self.o3d_vis.add_geometry(self.sim_mesh)
            self.coord_trans = torch.from_numpy(sRot.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()).float().cuda()

            self.o3d_vis.register_key_callback(32, self.pause_func) # space
            self.o3d_vis.register_key_callback(82, self.reset_func) # R
            self.o3d_vis.register_key_callback(76, self.record_func) # L
            self.o3d_vis.register_key_callback(84, self.next_func) # T
            self.o3d_vis.register_key_callback(75, self.hide_ref) # K
            
            self._video_queue_o3d = deque(maxlen=self.max_video_queue_size)
            rendering_out = self.cfg.get("rendering_output_root", osp.join("output", "renderings"))
            self._video_path_o3d = osp.join(rendering_out, f"{self.cfg_name}-%s-o3d.mp4")
            self._o3d_close_marker_path = osp.join(rendering_out, f"{self.cfg_name}-o3d-closed.marker")
            Path(self._o3d_close_marker_path).unlink(missing_ok=True)
            self.recording_state_change_o3d = False
            self._o3d_camera_initialized = False
            
            # if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            #     self.control = control = self.o3d_vis.get_view_control()
            #     control.unset_constant_z_far()
            #     control.unset_constant_z_near()
            #     control.set_up(np.array([0, 0, 1]))
            #     control.set_front(np.array([1, 0, 0]))
            #     control.set_zoom(0.001)

    def _o3d_center_from_verts(self, verts):
        pts = np.asarray(verts)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return None

        finite_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite_mask]
        if pts.shape[0] == 0:
            return None

        low = np.nanpercentile(pts, 2.0, axis=0)
        high = np.nanpercentile(pts, 98.0, axis=0)
        center = np.nanmedian(pts, axis=0)
        height = max(float(high[2] - low[2]), 1e-4)
        ratio = float(np.clip(self.o3d_camera_lookat_height_ratio, 0.20, 0.70))
        center[2] = low[2] + height * ratio + self.o3d_camera_lookat_z_offset
        if not np.all(np.isfinite(center)):
            return None
        return center

    def _update_o3d_camera_center(self, center):
        center = np.asarray(center, dtype=np.float64)
        if self._o3d_target_center is None:
            self._o3d_target_center = center
        else:
            delta = float(np.linalg.norm(center - self._o3d_target_center))
            alpha = float(np.clip(self.o3d_camera_follow_alpha, 0.01, 1.0))
            if delta > 2.0:
                self._o3d_target_center = center
            else:
                self._o3d_target_center = (1.0 - alpha) * self._o3d_target_center + alpha * center
        self._o3d_last_center = self._o3d_target_center
        return self._o3d_target_center

    def _o3d_camera_elevation(self):
        elevation = float(self.o3d_camera_elevation_deg)
        if self.o3d_camera_mode == "topdown":
            return float(np.clip(elevation, 45.0, 89.0))
        return float(np.clip(elevation, 5.0, 45.0))

    def _o3d_camera_vectors(self):
        yaw = np.radians(float(self._o3d_rotation_angle))
        elevation = np.radians(self._o3d_camera_elevation())
        horizontal = np.array([np.sin(yaw), -np.cos(yaw), 0.0], dtype=np.float64)
        camera_dir = horizontal * np.cos(elevation) + np.array([0.0, 0.0, np.sin(elevation)], dtype=np.float64)
        camera_dir /= max(np.linalg.norm(camera_dir), 1e-8)
        front = camera_dir

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = world_up - front * float(np.dot(world_up, front))
        if np.linalg.norm(up) < 1e-6:
            up = np.array([-np.cos(yaw), -np.sin(yaw), 0.0], dtype=np.float64)
        up /= max(np.linalg.norm(up), 1e-8)
        return front, up

    def _apply_o3d_camera(self, center=None):
        if not hasattr(self, "o3d_vis") or self.o3d_vis is None:
            return False
        if center is None:
            center = self._o3d_last_center
        if center is None:
            return False

        center = np.asarray(center, dtype=np.float64)
        if not np.all(np.isfinite(center)):
            return False

        control = self.o3d_vis.get_view_control()
        for method_name in ("unset_constant_z_far", "unset_constant_z_near"):
            try:
                getattr(control, method_name)()
            except Exception:
                pass

        front, up = self._o3d_camera_vectors()
        try:
            control.set_lookat(center.tolist())
            control.set_front(front.tolist())
            control.set_up(up.tolist())
            control.set_zoom(float(self.o3d_camera_zoom))
            return True
        except Exception as exc:
            print(f"[WARN] Failed to configure O3D camera: {exc}")
            return False

    def _configure_o3d_camera(self, verts):
        center = self._o3d_center_from_verts(verts)
        if center is None:
            return
        center = self._update_o3d_camera_center(center)
        if getattr(self, "_o3d_camera_initialized", False):
            return
        if self._apply_o3d_camera(center):
            self._o3d_camera_initialized = True

    def _wait_for_o3d_close(self):
        if not hasattr(self, "o3d_vis") or self.o3d_vis is None:
            return
        timeout_s = float(self.cfg.get("o3d_close_timeout_s", 0.0))
        if timeout_s <= 0:
            print("============ O3D recording saved. Close the O3D window to continue (no timeout). ============")
            while True:
                try:
                    alive = self.o3d_vis.poll_events()
                    self.o3d_vis.update_renderer()
                except Exception:
                    break
                if alive is False:
                    break
                time.sleep(0.2)
        else:
            print(f"============ O3D recording saved. Close the O3D window to continue (auto-close in {timeout_s:.0f}s). ============")
            start = time.time()
            while time.time() - start < timeout_s:
                try:
                    alive = self.o3d_vis.poll_events()
                    self.o3d_vis.update_renderer()
                except Exception:
                    break
                if alive is False:
                    break
                time.sleep(0.2)
            else:
                print(f"============ O3D window auto-closing after {timeout_s:.0f}s timeout ============")

    def render(self, sync_frame_time = False, i = 0):
        super().render(sync_frame_time=sync_frame_time)
        
        if self.viewer_o3d and getattr(self, "control_i", 0) == 0:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                assert(self._rigid_body_rot.shape[0] == 1)
                if self._has_upright_start:
                    body_quat = self._rigid_body_rot
                    root_trans = self._rigid_body_pos[:, 0, :]
                    
                    if self.vis_ref and len(self.ref_motion_cache.get('dof_pos', [])) == self.num_envs:
                        ref_body_quat = self.ref_motion_cache.get('rb_rot', torch.zeros(1, 24, 4))
                        ref_root_trans = self.ref_motion_cache.get('root_pos', torch.zeros(1, 3))
                        
                        body_quat = torch.cat([body_quat, ref_body_quat])
                        root_trans = torch.cat([root_trans, ref_root_trans])
                        
                    N = body_quat.shape[0]
                    offset = self.skeleton_trees[0].local_translation[0].cuda()
                    root_trans_offset = root_trans - offset
                    
                    pose_quat = (sRot.from_quat(body_quat.reshape(-1, 4).numpy()) * self.pre_rot).as_quat().reshape(N, -1, 4)
                    new_sk_state = SkeletonState.from_rotation_and_root_translation(self.skeleton_trees[0], torch.from_numpy(pose_quat), root_trans.cpu(), is_local=False)
                    local_rot = new_sk_state.local_rotation
                    pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_rotvec().reshape(N, -1, 3)
                    pose_aa = torch.from_numpy(pose_aa[:, self.mujoco_2_smpl, :].reshape(N, -1)).cuda()
                else:
                    dof_pos = self._dof_pos
                    root_trans = self._rigid_body_pos[:, 0, :]
                    root_rot = self._rigid_body_rot[:, 0, :]
                    pose_aa = torch.cat([torch_utils.quat_to_exp_map(root_rot), dof_pos], dim=1).reshape(1, -1)

                    if self.vis_ref and len(self.ref_motion_cache.get('dof_pos', [])) == self.num_envs:
                        ref_dof_pos = self.ref_motion_cache.get('dof_pos')
                        ref_root_rot = self.ref_motion_cache.get('rb_rot', torch.zeros(1, 24, 4))[:, 0, :]
                        ref_root_trans = self.ref_motion_cache.get('root_pos', torch.zeros(1, 3))
                        
                        ref_pose_aa = torch.cat([torch_utils.quat_to_exp_map(ref_root_rot), ref_dof_pos], dim=1)
                        
                        pose_aa = torch.cat([pose_aa, ref_pose_aa])
                        root_trans = torch.cat([root_trans, ref_root_trans])
                    N = pose_aa.shape[0]
                    offset = self.skeleton_trees[0].local_translation[0].cuda()
                    root_trans_offset = root_trans - offset
                    pose_aa = pose_aa.view(N, -1, 3)[:, self.mujoco_2_smpl, :]


                with torch.no_grad():
                    verts, joints = self.mesh_parser.get_joints_verts(pose=pose_aa, th_trans=root_trans_offset.cuda())
                    
            sim_verts = verts.numpy()[0]
            self.sim_mesh.vertices = o3d.utility.Vector3dVector(sim_verts)
            self._configure_o3d_camera(sim_verts)
            if N > 1:
                ref_verts = verts.numpy()[1]
                if not flags.show_traj:
                    ref_verts[..., 0] += 2
                self.ref_mesh.vertices = o3d.utility.Vector3dVector(ref_verts)
                    
            self.sim_mesh.compute_vertex_normals()
            self.o3d_vis.update_geometry(self.sim_mesh)
            if N > 1:
                self.o3d_vis.update_geometry(self.ref_mesh)

            self.sim_mesh.compute_vertex_normals()
            if self.vis_ref:
                self.ref_mesh.compute_vertex_normals()
            alive = self.o3d_vis.poll_events()
            self.o3d_vis.update_renderer()

            # Interactive mode: check if user closed the O3D window
            if getattr(self, "_o3d_interactive_mode", False):
                if alive is False:
                    print("============ O3D window closed by user, proceeding... ============")
                    try:
                        Path(self._o3d_close_marker_path).write_text("closed\n", encoding="utf-8")
                    except Exception:
                        pass
                    # Gracefully stop O3D rendering instead of killing the process.
                    # This allows the simulation loop to finish normally and write the state file.
                    try:
                        self.o3d_vis.destroy_window()
                    except Exception:
                        pass
                    self.o3d_vis = None
                    self.viewer_o3d = False
                    self._o3d_interactive_mode = False
                    print("============ O3D window closed — simulation continues to completion ============")

            if self.recording_state_change_o3d:
                if not self.recording:
                    if len(self._video_queue_o3d) == 0:
                        print("============ O3D recording finished with no frames ============")
                    else:
                        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                        curr_video_file_name = self._video_path_o3d % curr_date_time
                        Path(curr_video_file_name).parent.mkdir(parents=True, exist_ok=True)
                        fps = 30
                        writer = imageio.get_writer(curr_video_file_name, fps=fps, macro_block_size=None)
                        frames = list(self._video_queue_o3d)
                        height, width = frames[0].shape[:2]
                        height = max(2, height - (height % 2))
                        width = max(2, width - (width % 2))

                        try:
                            for frame in tqdm(frames):
                                frame = np.asarray(frame)
                                if frame.ndim == 2:
                                    frame = np.repeat(frame[..., None], 3, axis=2)
                                if frame.shape[2] > 3:
                                    frame = frame[:, :, :3]
                                if frame.shape[0] != height or frame.shape[1] != width:
                                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                                writer.append_data(frame[:height, :width, :])
                        finally:
                            writer.close()
                        self._video_queue_o3d = deque(maxlen=self.max_video_queue_size)

                        print(f"============ Video finished writing O3D {curr_video_file_name}============")
                    if getattr(self, "wait_for_o3d_close_after_record", False):
                        self._pending_o3d_auto_record_exit = False
                        self._o3d_interactive_mode = True
                        # Give the parent process time to detect the state file
                        # (needs 3 stable checks at 1-second intervals)
                        import time as _time
                        _time.sleep(3.5)
                        print("============ O3D window stays open — close it to continue to next step. ============")
                    elif self._pending_o3d_auto_record_exit:
                        self._pending_o3d_auto_record_exit = False
                        # Brief delay to let parent detect the state file before we exit
                        import time as _time
                        _time.sleep(1.0)
                        self._cleanup_and_exit()
                else:
                    print(f"============ Writing video O3D ============")
                    
                self.recording_state_change_o3d = False
                
            if self.recording:
                # Auto-rotate camera around the subject for comprehensive views
                if getattr(self, "o3d_auto_rotate", False):
                    self._o3d_rotation_angle += self.o3d_rotation_speed
                    if self._o3d_rotation_angle >= 360.0:
                        self._o3d_rotation_angle -= 360.0
                if getattr(self, "o3d_camera_follow", True) or getattr(self, "o3d_auto_rotate", False):
                    self._apply_o3d_camera()
                rgb = self.o3d_vis.capture_screen_float_buffer()
                rgb = (np.asarray(rgb) * 255).astype(np.uint8)
                self._video_queue_o3d.append(rgb)

    def _load_motion(self, motion_train_file, motion_test_file=[]):
        assert (self._dof_offsets[-1] == self.num_dof)
        fix_height_mode = _resolve_fix_height_mode(self.cfg)

        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            motion_lib_cfg = EasyDict({
                "motion_file": motion_train_file,
                "device": torch.device("cpu"),
                "fix_height": fix_height_mode,
                "min_length": self._min_motion_len,
                "max_length": -1,
                "im_eval": flags.im_eval,
                "multi_thread": not self.cfg.disable_multiprocessing ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "device": self.device,
                "step_dt": self.dt,
            })
            motion_eval_file = motion_train_file
            self._motion_train_lib = MotionLibSMPL(motion_lib_cfg)
            motion_lib_cfg.im_eval = True
            self._motion_eval_lib = MotionLibSMPL(motion_lib_cfg)

            self._motion_lib = self._motion_train_lib
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(),
                                          limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions),
                                          max_len=-1 if flags.test else self.max_len, start_idx=self.start_idx)
        elif self.humanoid_type in ['h1', 'g1']:
            motion_lib_cfg = EasyDict({
                "motion_file": motion_train_file,
                "device": torch.device("cpu"),
                "fix_height": fix_height_mode,
                "min_length": self._min_motion_len,
                "max_length": self.max_len,
                "im_eval": flags.im_eval,
                "multi_thread": not self.cfg.disable_multiprocessing ,
                "smpl_type": self.humanoid_type,
                "randomrize_heading": True,
                "device": self.device,
                "robot": self.cfg.robot,
                "step_dt": self.dt,
            })
            motion_eval_file = motion_train_file
            self._motion_train_lib = MotionLibReal(motion_lib_cfg)
            self._motion_eval_lib = MotionLibReal(motion_lib_cfg)

            self._motion_lib = self._motion_train_lib
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=(not flags.test) and (not self.seq_motions), max_len=-1 if flags.test else self.max_len)
            
        else:
            self._motion_lib = MotionLib(motion_file=motion_train_file, dof_body_ids=self._dof_body_ids, dof_offsets=self._dof_offsets, device=self.device)

        return

    def resample_motions(self):
        # self.gym.destroy_sim(self.sim)
        # del self.sim
        # if not self.headless:
        #     self.gym.destroy_viewer(self.viewer)
        # self.create_sim()
        # self.gym.prepare_sim(self.sim)
        # self.create_viewer()
        # self._setup_tensors()

        print("Partial solution, only resample motions...")
        # if self.hard_negative:
            # self._motion_lib.update_sampling_weight()

        if flags.test:
            self.forward_motion_samples()
        else:
            self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, limb_weights=self.humanoid_limb_and_weights.cpu(), gender_betas=self.humanoid_shapes.cpu(), random_sample=(not flags.test) and (not self.seq_motions),
                                          max_len=-1 if flags.test else self.max_len)  # For now, only need to sample motions since there are only 400 hmanoids

            # self.reset() #
            # print("Reasmpling and resett!!!.")

            time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
            root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids, time)
            self._global_offset[:, :2] = self._humanoid_root_states[:, :2] - root_res['root_pos'][:, :2]
            self.reset()


    def get_motion_lengths(self):
        return self._motion_lib.get_motion_lengths()

    def _record_states(self):
        super()._record_states()
        self.state_record['ref_body_pos_subset'].append(self.ref_body_pos_subset.cpu().clone())
        self.state_record['ref_body_pos_full'].append(self.ref_body_pos.cpu().clone())
        # self.state_record['ref_dof_pos'].append(self.ref_dof_pos.cpu().clone())

    def _write_states_to_file(self, file_name):
        self.state_record['skeleton_trees'] = self.skeleton_trees
        self.state_record['humanoid_betas'] = self.humanoid_shapes
        print(f"Dumping states into {file_name}")

        if len(self.state_record.get('progress', [])) == 0:
            print("[WARN] state_record is empty — no states have been recorded. Skipping state file write.")
            return

        progress = torch.stack(self.state_record['progress'], dim=1)
        progress_diff = torch.cat([progress, -10 * torch.ones(progress.shape[0], 1).to(progress)], dim=-1)

        diff = torch.abs(progress_diff[:, :-1] - progress_diff[:, 1:])
        split_idx = torch.nonzero(diff > 1)
        split_idx[:, 1] += 1
        data_to_dump = {k: torch.stack(v) for k, v in self.state_record.items() if k not in ['skeleton_trees', 'humanoid_betas', "progress"]}
        fps = int(round(1.0 / self.dt))
        motion_dict_dump = {}
        num_for_this_humanoid = 0
        curr_humanoid_index = 0

        for idx in range(len(split_idx)):
            split_info = split_idx[idx]
            humanoid_index = split_info[0]

            if humanoid_index != curr_humanoid_index:
                num_for_this_humanoid = 0
                curr_humanoid_index = humanoid_index

            if num_for_this_humanoid == 0:
                start = 0
            else:
                start = split_idx[idx - 1][-1]

            end = split_idx[idx][-1]

            dof_pos_seg = data_to_dump['dof_pos'][start:end, humanoid_index]
            B, H = dof_pos_seg.shape
            root_states_seg = data_to_dump['root_states'][start:end, humanoid_index]
            if self.humanoid_type in ['h1', 'g1' ]:
                motion_dump = {
                    "skeleton_tree": self.state_record['skeleton_trees'][humanoid_index].to_dict(),
                    "trans": root_states_seg[:, :3],
                    "root_states_seg": root_states_seg,
                    "dof_pos": dof_pos_seg,
                }
            else:
                body_quat = torch.cat([root_states_seg[:, None, 3:7], torch_utils.exp_map_to_quat(dof_pos_seg.reshape(B, -1, 3))], dim=1)

                motion_dump = {
                    "skeleton_tree": self.state_record['skeleton_trees'][humanoid_index].to_dict(),
                    "body_quat": body_quat,
                    "trans": root_states_seg[:, :3],
                    "root_states_seg": root_states_seg,
                    "dof_pos": dof_pos_seg,
                }
                
            motion_dump['fps'] = fps
            motion_dump['betas'] = self.humanoid_shapes[humanoid_index].detach().cpu().numpy()
            motion_dump.update({k: v[start:end, humanoid_index] for k, v in data_to_dump.items() if k not in ['dof_pos', 'root_states', 'skeleton_trees', 'humanoid_betas', "progress"]})
            motion_dict_dump[f"{humanoid_index}_{num_for_this_humanoid}"] = motion_dump
            num_for_this_humanoid += 1
        joblib.dump(motion_dict_dump, file_name)
        # Ensure the state file is flushed to disk so the parent process can detect it
        try:
            import os as _os
            with open(file_name, 'rb') as _f:
                _os.fsync(_f.fileno())
        except Exception:
            pass
        print(f"============ State file written and synced: {file_name} ============")
        self.state_record = defaultdict(list)

    def begin_seq_motion_samples(self):
        # For evaluation
        self.start_idx = 0
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=False, start_idx=self.start_idx)
        self.reset()

    def forward_motion_samples(self):
        self.start_idx += self.num_envs
        self._motion_lib.load_motions(skeleton_trees=self.skeleton_trees, gender_betas=self.humanoid_shapes.cpu(), limb_weights=self.humanoid_limb_and_weights.cpu(), random_sample=False, start_idx=self.start_idx)
        self.reset()

    # Disabled.
    # def get_self_obs_size(self):
    #     if self.obs_v == 4:
    #         return self._num_self_obs * self.past_track_steps
    #     else:
    #         return self._num_self_obs

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            if self.obs_v == 1:
                obs_size = len(self._track_bodies) * self._num_traj_samples * 15
            elif self.obs_v == 2:  # + dofdiff
                obs_size = len(self._track_bodies) * self._num_traj_samples * 15
                obs_size += (len(self._track_bodies) - 1) * self._num_traj_samples * 3
            elif self.obs_v == 3:  # reduced number
                obs_size = len(self._track_bodies) * self._num_traj_samples * 9
            elif self.obs_v == 4:  # 10 steps + v6
                
                # obs_size = len(self._track_bodies) * self._num_traj_samples * 15 * 5
                obs_size = len(self._track_bodies) * 15
                obs_size += len(self._track_bodies) * self._num_traj_samples * 9
                obs_size *= self.past_track_steps
                
            elif self.obs_v == 5:  # one hot vector for type of motions
                obs_size = len(self._track_bodies) * self._num_traj_samples * 24 + 30 # Hard coded. 
            elif self.obs_v == 6:  # local+ dof + pos (not diff)
                obs_size = len(self._track_bodies) * self._num_traj_samples * 24

            elif self.obs_v == 7:  # local+ dof + pos (not diff)
                obs_size = len(self._track_bodies) * self._num_traj_samples * 9  # linear position + velocity

            elif self.obs_v == 8:  # local+ dof + pos (not diff) + vel (no diff). 
                obs_size = len(self._track_bodies) * 15
                obs_size += len(self._track_bodies) * self._num_traj_samples * 15

            elif self.obs_v == 9:  # local+ dof + pos (not diff) + vel (no diff). 
                obs_size = len(self._track_bodies) * self._num_traj_samples * 24
                obs_size -= (len(self._track_bodies) - 1) * self._num_traj_samples * 6


        return obs_size

    def get_task_obs_size_detail(self):
        task_obs_detail = OrderedDict()
        task_obs_detail['target'] = self.get_task_obs_size()
        task_obs_detail['fut_tracks'] = self._fut_tracks
        task_obs_detail['num_traj_samples'] = self._num_traj_samples
        task_obs_detail['obs_v'] = self.obs_v
        task_obs_detail['track_bodies'] = self._track_bodies
        task_obs_detail['models_path'] = self.models_path

        # Dev
        task_obs_detail['num_prim'] = self.cfg['env'].get("num_prim", 2)
        task_obs_detail['training_prim'] = self.cfg['env'].get("training_prim", 1)
        task_obs_detail['actors_to_load'] = self.cfg['env'].get("actors_to_load", 2)
        task_obs_detail['has_lateral'] = self.cfg['env'].get("has_lateral", True)

        return task_obs_detail

    def _build_termination_heights(self):
        super()._build_termination_heights()
        termination_distance = self.cfg["env"].get("terminationDistance", 0.5)
        self._termination_distances = to_torch(np.array([termination_distance] * self.num_bodies), device=self.device)
        return

    def init_root_points(self):
        # For debugging purpose
        y = torch.tensor(np.linspace(-0.5, 0.5, 5), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(0, 1, 5), device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_root_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless or flags.server_mode):
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        if flags.add_proj:
            self._proj_handles = []
            self._load_proj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 0.0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, "traj_marker.urdf", asset_options)
        
        self._marker_asset_small = self.gym.load_asset(self.sim, asset_root, "traj_marker_small.urdf", asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless or flags.server_mode):
            self._build_marker(env_id, env_ptr)

        if flags.add_proj:
            self._build_proj(env_id, env_ptr)

        return
    
    def _update_marker(self):
        if flags.show_traj:
            
            motion_times = (self.progress_buf + 1) * self.dt + self._motion_start_times + self._motion_start_times_offset # + 1 for target. 
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
            
            self._marker_pos[:] = ref_rb_pos
            # self._marker_rotation[..., self._track_bodies_id, :] = ref_rb_rot[..., self._track_bodies_id, :]
            
            ## Only update the tracking points. 
            if flags.real_traj:
                self._marker_pos[:] = 1000
                
            self._marker_pos[..., self._track_bodies_id, :] = ref_rb_pos[..., self._track_bodies_id, :]

            if self._occl_training:
                self._marker_pos[self.random_occlu_idx] = 0

        else:
            self._marker_pos[:] = 1000

        # ######### Heading debug #######
        # points = self.init_root_points()
        # base_quat = self._rigid_body_rot[0, 0:1]
        # base_quat = remove_base_rot(base_quat)
        # heading_rot = torch_utils.calc_heading_quat(base_quat)
        # show_points = quat_apply(heading_rot.repeat(1, points.shape[0]).reshape(-1, 4), points) + (self._rigid_body_pos[0, 0:1]).unsqueeze(1)
        # self._marker_pos[:] = show_points[:, :self._marker_pos.shape[1]]
        # ######### Heading debug #######

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        for i in range(self._num_joints):
            # Giving hands smaller balls to indicate positions
            if self.humanoid_type in ['smplx'] and self._body_names_orig[i] in ["L_Wrist", "R_Wrist", "L_Index1", "L_Index2", "L_Index3","L_Middle1","L_Middle2","L_Middle3","L_Pinky1","L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3", "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3", "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",]:
                marker_handle = self.gym.create_actor(env_ptr, self._marker_asset_small, default_pose, "marker", self.num_envs + 10, 1, 0)    
            else:
                marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10, 1, 0)
            
            if i in self._track_bodies_id:
                self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
            else:
                self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 1.0))
            self._marker_handles[env_id].append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1:(1 + self._num_joints), :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rotation = self._marker_states[..., 3:7]

        self._marker_actor_ids = self._humanoid_actor_ids.unsqueeze(-1) + to_torch(self._marker_handles, dtype=torch.int32, device=self.device)
        self._marker_actor_ids = self._marker_actor_ids.flatten()

        return

    def _sample_time(self, motion_ids):
        # Motion imitation, no more blending and only sample at certain locations
        return self._motion_lib.sample_time_interval(motion_ids)
        # return self._motion_lib.sample_time(motion_ids)

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        # imitation task is resetted with the actions
        return

    def post_physics_step(self):
        super().post_physics_step()

        if self.auto_record_motion and (not self._auto_record_done):
            if (not self._auto_record_started) and self._motion_lib.num_motions() > 0:
                self._auto_record_num_steps = max(int(self._motion_lib.get_motion_num_steps(torch.tensor([0], device=self.device))[0].item()), 1)
                self._auto_record_frames = 0
                self.recording = True
                self.recording_state_change = True
                if self.viewer_o3d:
                    self.recording_state_change_o3d = True
                self._auto_record_started = True
                # Auto-calculate rotation speed for 2-3 full horizontal rotations
                if self.o3d_auto_rotate and self.o3d_rotation_speed <= 0:
                    total_degrees = 360.0 * self.o3d_num_rotations
                    self.o3d_rotation_speed = total_degrees / max(self._auto_record_num_steps, 1)
                print(f"============ Auto recording started for {self._auto_record_num_steps} sim steps ============")
                if self.o3d_auto_rotate:
                    print(
                        f"============ O3D auto-rotate: {self.o3d_rotation_speed:.1f} deg/frame = "
                        f"{self.o3d_num_rotations:.1f} rotations, camera={self.o3d_camera_mode}, "
                        f"elevation={self._o3d_camera_elevation():.1f} deg, "
                        f"follow_alpha={self.o3d_camera_follow_alpha:.2f}, floor={self.o3d_floor_style} ============"
                    )
            elif self._auto_record_started:
                self._auto_record_frames += 1
                if self._auto_record_frames >= self._auto_record_num_steps:
                    self.recording = False
                    self.recording_state_change = True
                    self._auto_record_done = True
                    if self.viewer_o3d:
                        self.recording_state_change_o3d = True
                        self._pending_o3d_auto_record_exit = self.auto_quit_after_record or self.wait_for_o3d_close_after_record
                        self._pending_auto_record_exit = False
                    else:
                        self._pending_auto_record_exit = self.auto_quit_after_record
                    print("============ Auto recording finished ============")
                    # Always finalize recording and write state file immediately.
                    # Previously this was skipped when viewer_o3d=True, relying on
                    # recording_state_change -> render() -> _write_states_to_file,
                    # but that indirect path was unreliable (state file never written).
                    self._finalize_recording_now()
                    self.recording_state_change = False  # Already handled above
                    if self._pending_auto_record_exit and not self.viewer_o3d:
                        self._cleanup_and_exit()
        
        if flags.im_eval:
            motion_times = (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset  # already has time + 1, so don't need to + 1 to get the target for "this frame"
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset)  # pass in the env_ids such that the motion is in synced.
            body_pos = self._rigid_body_pos
            self.extras['mpjpe'] = (body_pos - motion_res['rg_pos']).norm(dim=-1).mean(dim=-1)
            self.extras['body_pos'] = body_pos.cpu().numpy()
            self.extras['body_pos_gt'] = motion_res['rg_pos'].cpu().numpy()

            #### Dumping dataset
            if self.collect_dataset:
                self.extras['obs_buf'] = self.obs_buf_t.copy()  # n, 945
                self.extras['actions'] = self.actions.cpu().numpy()  # n, 69
                self.extras['clean_actions'] = self.clean_actions.cpu().numpy()
                self.extras['reset_buf'] = self.reset_buf.cpu().numpy()  # n

            
                self.obs_buf_t = self.obs_buf.cpu().numpy() # update to next time step

        return

    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        
        self.self_obs_buf[env_ids] = self_obs

        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
        else:
            obs = self_obs
            
        if self.add_obs_noise and not flags.test:
            obs = obs + torch.randn_like(obs) * 0.1

        if self.obs_v == 4:
            # Double sub will return a copy.
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs
        
        return obs

    def _compute_task_obs(self, env_ids=None, save_buffer = True):
        if (env_ids is None):
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]

        curr_gender_betas = self.humanoid_shapes[env_ids]
        
        if self._fut_tracks:
            time_steps = self._num_traj_samples
            B = env_ids.shape[0]
            time_internals = torch.arange(time_steps).to(self.device).repeat(B).view(-1, time_steps) * self._traj_sample_timestep
            motion_times_steps = ((self.progress_buf[env_ids, None] + 1) * self.dt + time_internals + self._motion_start_times[env_ids, None] + self._motion_start_times_offset[env_ids, None]).flatten()  # Next frame, so +1
            env_ids_steps = self._sampled_motion_ids[env_ids].repeat_interleave(time_steps)
            motion_res = self._get_state_from_motionlib_cache(env_ids_steps, motion_times_steps, self._global_offset[env_ids].repeat_interleave(time_steps, dim=0).view(-1, 3))  # pass in the env_ids such that the motion is in synced.

        else:
            motion_times = (self.progress_buf[env_ids] + 1) * self.dt + self._motion_start_times[env_ids] + self._motion_start_times_offset[env_ids]  # Next frame, so +1
            time_steps = 1
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids])  # pass in the env_ids such that the motion is in synced.

        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        body_pos_subset = body_pos[..., self._track_bodies_id, :]
        body_rot_subset = body_rot[..., self._track_bodies_id, :]
        body_vel_subset = body_vel[..., self._track_bodies_id, :]
        body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

        ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
        ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
        ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
        ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]

        if self.obs_v == 1 :
            obs = compute_imitation_observations(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)

        elif self.obs_v == 2:
            ref_dof_pos_subset = ref_dof_pos.reshape(-1, len(self._dof_names), 3)[..., self._track_bodies_id[1:] - 1, :]  # Remove root from dof dim
            dof_pos_subset = self._dof_pos[env_ids].reshape(-1, len(self._dof_names), 3)[..., self._track_bodies_id[1:] - 1, :]
            obs = compute_imitation_observations_v2(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, dof_pos_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, ref_dof_pos_subset, time_steps, self._has_upright_start)
        elif self.obs_v == 3:
            obs = compute_imitation_observations_v3(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)
        elif self.obs_v == 4 or self.obs_v == 5 or self.obs_v == 6 or self.obs_v == 8 or self.obs_v == 9:

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_rb_rot_subset[zeros_subset, 1:] = body_rot_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
                ref_body_ang_vel_subset[zeros_subset, :] = body_ang_vel_subset[zeros_subset, :]
                self._point_goal[env_ids] = distance

                far_distance = self.far_distance  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]

            if self._occl_training:
                # ranomly occlude some of the body parts
                random_occlu_idx = self.random_occlu_idx[env_ids]
                ref_rb_pos_subset[random_occlu_idx] = body_pos_subset[random_occlu_idx]
                ref_rb_rot_subset[random_occlu_idx] = body_rot_subset[random_occlu_idx]
                ref_body_vel_subset[random_occlu_idx] = body_vel_subset[random_occlu_idx]
                ref_body_ang_vel_subset[random_occlu_idx] = body_ang_vel_subset[random_occlu_idx]

            if self.obs_v == 4 or self.obs_v == 6:
                obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)
                
                # obs[:, -1] = env_ids.clone().float(); print('debugging')
                # obs[:, -2] = self.progress_buf[env_ids].clone().float(); print('debugging')
                
            elif self.obs_v == 5:
                obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)
                one_hots = self._motion_lib.one_hot_motions[env_ids]
                obs = torch.cat([obs, one_hots], dim=-1)
                
            elif self.obs_v == 8:
                obs = compute_imitation_observations_v8(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, time_steps, self._has_upright_start)
            elif self.obs_v == 9:
                ref_root_vel_subset = ref_body_vel_subset[:, 0]
                ref_root_ang_vel_subset =ref_body_ang_vel_subset[:, 0]
                obs = compute_imitation_observations_v9(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_root_vel_subset, ref_root_ang_vel_subset, time_steps, self._has_upright_start)
            
            if self._fut_tracks_dropout and not flags.test:
                dropout_rate = 0.1
                curr_num_envs = env_ids.shape[0]
                obs = obs.view(curr_num_envs, self._num_traj_samples, -1)
                mask = torch.rand(curr_num_envs, self._num_traj_samples) < dropout_rate
                obs[mask, :] = 0
                obs = obs.view(curr_num_envs, -1)
                
        elif self.obs_v == 7:

            if self.zero_out_far:
                close_distance = self.close_distance
                distance = torch.norm(root_pos - ref_rb_pos_subset[..., 0, :], dim=-1)

                zeros_subset = distance > close_distance
                ref_rb_pos_subset[zeros_subset, 1:] = body_pos_subset[zeros_subset, 1:]
                ref_body_vel_subset[zeros_subset, :] = body_vel_subset[zeros_subset, :]
                self._point_goal[env_ids] = distance

                far_distance = self.far_distance  # does not seem to need this in particular...
                vector_zero_subset = distance > far_distance  # > 5 meters, it become just a direction
                ref_rb_pos_subset[vector_zero_subset, 0] = ((ref_rb_pos_subset[vector_zero_subset, 0] - body_pos_subset[vector_zero_subset, 0]) / distance[vector_zero_subset, None] * far_distance) + body_pos_subset[vector_zero_subset, 0]

            if self._occl_training:
                # ranomly occlude some of the body parts
                random_occlu_idx = self.random_occlu_idx[env_ids]
                ref_rb_pos_subset[random_occlu_idx] = body_pos_subset[random_occlu_idx]
                ref_rb_rot_subset[random_occlu_idx] = body_rot_subset[random_occlu_idx]
            
            obs = compute_imitation_observations_v7(root_pos, root_rot, body_pos_subset, body_vel_subset, ref_rb_pos_subset, ref_body_vel_subset, time_steps, self._has_upright_start)
            
        if save_buffer:
            if self._fut_tracks:
                self.ref_body_pos[env_ids] = ref_rb_pos[..., 0, :, :]
                self.ref_body_vel[env_ids] = ref_body_vel[..., 0, :, :]
                self.ref_body_rot[env_ids] = ref_rb_rot[..., 0, :, :]
                self.ref_body_pos_subset[env_ids] = ref_rb_pos_subset[..., 0, :, :]
                self.ref_dof_pos[env_ids] = ref_dof_pos[..., 0,  :]
                
            else:
                self.ref_body_pos[env_ids] = ref_rb_pos
                self.ref_body_vel[env_ids] = ref_body_vel
                self.ref_body_rot[env_ids] = ref_rb_rot
                self.ref_body_pos_subset[env_ids] = ref_rb_pos_subset
                self.ref_dof_pos[env_ids] = ref_dof_pos
        
        
        return obs

    def _compute_reward(self, actions):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset  # reward is computed after phsycis step, and progress_buf is already updated for next time step.

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset) 

        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

        if self.zero_out_far:
            transition_distance = 0.25
            distance = torch.norm(root_pos - ref_root_pos, dim=-1)

            zeros_subset = distance > transition_distance  # For those that are outside, no imitation reward
            self.reward_raw = torch.zeros((self.num_envs, 4)).to(self.device)

            # self.rew_buf, self.reward_raw[:, 0] = compute_location_reward(root_pos, ref_rb_pos[..., 0, :])
            self.rew_buf, self.reward_raw[:, 0] = compute_point_goal_reward(self._point_goal, distance)

            im_reward, im_reward_raw = compute_imitation_reward(root_pos[~zeros_subset, :], root_rot[~zeros_subset, :], body_pos[~zeros_subset, :], body_rot[~zeros_subset, :], body_vel[~zeros_subset, :], body_ang_vel[~zeros_subset, :], ref_rb_pos[~zeros_subset, :], ref_rb_rot[~zeros_subset, :],
                                                                ref_body_vel[~zeros_subset, :], ref_body_ang_vel[~zeros_subset, :], self.reward_specs)

            # self.rew_buf, self.reward_raw = self.rew_buf * 0.5, self.reward_raw * 0.5 # Half the reward for the location reward
            self.rew_buf[~zeros_subset] = self.rew_buf[~zeros_subset] + im_reward * 0.5  # for those are inside, add imitation reward
            self.reward_raw[~zeros_subset, :4] = self.reward_raw[~zeros_subset, :4] + im_reward_raw * 0.5

            # local_rwd, _ = compute_location_reward(root_pos, ref_rb_pos[:, ..., 0, :])
            # im_rwd, _ = compute_imitation_reward(
            #         root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel,
            #         ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel,
            #         self.reward_specs)
            # print(local_rwd, im_rwd)

        else:
            if self._full_body_reward:
                if self.humanoid_type in ['h1', 'g1']:
                    extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos_in_parent.reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
                    body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
                    body_rot_extend = torch.cat([body_rot, body_rot[:, self.extend_body_parent_ids]], dim=1)
                    ref_rb_pos_extend = torch.cat([ref_rb_pos, motion_res["rg_pos_t"][:, self.num_bodies:]], dim = 1)
                    ref_rb_rot_extend = torch.cat([ref_rb_rot, motion_res["rg_rot_t"][:, self.num_bodies:]], dim = 1)
                    
                    self.rew_buf[:], self.reward_raw = compute_imitation_reward(root_pos, root_rot, body_pos_extend, body_rot_extend, body_vel, body_ang_vel, ref_rb_pos_extend, ref_rb_rot_extend, ref_body_vel, ref_body_ang_vel, self.reward_specs)
                else:
                    self.rew_buf[:], self.reward_raw = compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel, self.reward_specs)
            else:
                body_pos_subset = body_pos[..., self._track_bodies_id, :]
                body_rot_subset = body_rot[..., self._track_bodies_id, :]
                body_vel_subset = body_vel[..., self._track_bodies_id, :]
                body_ang_vel_subset = body_ang_vel[..., self._track_bodies_id, :]

                ref_rb_pos_subset = ref_rb_pos[..., self._track_bodies_id, :]
                ref_rb_rot_subset = ref_rb_rot[..., self._track_bodies_id, :]
                ref_body_vel_subset = ref_body_vel[..., self._track_bodies_id, :]
                ref_body_ang_vel_subset = ref_body_ang_vel[..., self._track_bodies_id, :]
                self.rew_buf[:], self.reward_raw = compute_imitation_reward(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, self.reward_specs)

        # print(self.dof_force_tensor.abs().max())
        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
            # power_reward = -0.00005 * (power ** 2)
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)
        
        return
    
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if self.collect_dataset:
            self.obs_buf_t = self.obs_buf.cpu().numpy() # first time step update

    def _reset_ref_state_init(self, env_ids):
        self._motion_start_times_offset[env_ids] = 0  # Reset the motion time offsets
        self._global_offset[env_ids] = 0  # Reset the global offset when resampling.
        # self._global_offset[:, 0], self._global_offset[:, 1] = self.start_pos_x[:self.num_envs], self.start_pos_y[:self.num_envs]

        self._cycle_counter[env_ids] = 0
        super()._reset_ref_state_init(env_ids)  # This function does not use the offset
        # self._motion_lib.update_sampling_history(env_ids)

        if self.obs_v == 4:
            self.obs_buf[env_ids] = 0
        if self.zero_out_far and self.zero_out_far_train:
            # if self.zero_out_far and not flags.test:
            # Moving the start position to a random location
            # env_ids_pick = env_ids
            env_ids_pick = env_ids[torch.arange(env_ids.shape[0]).long()]  #  All far away start. 
            max_distance = 5
            rand_distance = torch.sqrt(torch.rand(env_ids_pick.shape[0]).to(self.device)) * max_distance

            rand_angle = torch.rand(env_ids_pick.shape[0]).to(self.device) * np.pi * 2

            self._global_offset[env_ids_pick, 0] = torch.cos(rand_angle) * rand_distance
            self._global_offset[env_ids_pick, 1] = torch.sin(rand_angle) * rand_distance

            # self._global_offset[env_ids]
            self._cycle_counter[env_ids_pick] = self._zero_out_far_steps

        return

    def _get_state_from_motionlib_cache(self, motion_ids, motion_times, offset=None):
        ## Cache the motion + offset
        if offset is None  or not "motion_ids" in self.ref_motion_cache or self.ref_motion_cache['offset'] is None or len(self.ref_motion_cache['motion_ids']) != len(motion_ids) or len(self.ref_motion_cache['offset']) != len(offset) \
            or  (self.ref_motion_cache['motion_ids'] - motion_ids).abs().sum() + (self.ref_motion_cache['motion_times'] - motion_times).abs().sum() + (self.ref_motion_cache['offset'] - offset).abs().sum() > 0 :
            self.ref_motion_cache['motion_ids'] = motion_ids.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['motion_times'] = motion_times.clone()  # need to clone; otherwise will be overriden
            self.ref_motion_cache['offset'] = offset.clone() if not offset is None else None
        else:
            return self.ref_motion_cache
        
        motion_res = self._motion_lib.get_motion_state(motion_ids, motion_times, offset=offset)

        self.ref_motion_cache.update(motion_res)

        return self.ref_motion_cache

    def _sample_ref_state(self, env_ids):
        num_envs = env_ids.shape[0]

        if (self._state_init == HumanoidAMP.StateInit.Random or self._state_init == HumanoidAMP.StateInit.Hybrid):
            motion_times = self._sample_time(self._sampled_motion_ids[env_ids])
        elif (self._state_init == HumanoidAMP.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert (False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        if flags.test:
            motion_times[:] = 0
        
        if self.humanoid_type in ['h1', 'g1',"smpl", "smplh", "smplx"] :
            motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids[env_ids], motion_times, self._global_offset[env_ids])
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(self._sampled_motion_ids[env_ids], motion_times)
            rb_pos, rb_rot = None, None
            
        return self._sampled_motion_ids[env_ids], motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel

    def _hack_motion_sync(self):
        if (not hasattr(self, "_hack_motion_time")):
            self._hack_motion_time = self._motion_start_times + self._motion_start_times_offset

        num_motions = self._motion_lib.num_motions()
        motion_ids = np.arange(self.num_envs, dtype=np.int)
        motion_ids = np.mod(motion_ids, num_motions)
        motion_ids = torch.from_numpy(motion_ids).to(self.device)
        # motion_ids[:] = 2
        motion_times = self._hack_motion_time
        if self.humanoid_type in ['h1',"smpl", "smplh", "smplx"] :
            motion_res = self._get_state_from_motionlib_cache(motion_ids, motion_times, self._global_offset)
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

            root_pos[..., -1] += 0.03  # ALways slightly above the ground to avoid issue
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
                = self._motion_lib.get_motion_state(motion_ids, motion_times)
            rb_pos, rb_rot = None, None

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rigid_body_pos=rb_pos,
            rigid_body_rot=rb_rot,
            rigid_body_vel=body_vel,
            rigid_body_ang_vel=body_ang_vel,
        )

        self._reset_env_tensors(env_ids)
        motion_fps = self._motion_lib._motion_fps[0]

        motion_dur = self._motion_lib._motion_lengths[0]
        if not self.paused:
            self._hack_motion_time = (self._hack_motion_time + self._motion_sync_dt)  # since the simulation is double
        else:
            pass

        # self.progress_buf[:] = (self._hack_motion_time *  2* motion_fps).long() # /2 is for simulation double speed...

        return

    def _update_cycle_count(self):
        self._cycle_counter -= 1
        self._cycle_counter = torch.clamp_min(self._cycle_counter, 0)
        return

    def _update_occl_training(self):
        occu_training = torch.ones([self.num_envs, len(self._track_bodies)], device=self.device) * self._occl_training_prob
        random_occlu_idx = torch.bernoulli(occu_training).bool()
        random_occlu_idx[:, 0] = False

        self.random_occlu_count[random_occlu_idx] = torch.randint(30, 60, self.random_occlu_count[random_occlu_idx].shape).to(self.device)
        self.random_occlu_count -= 1
        self.random_occlu_count = torch.clamp_min(self.random_occlu_count, 0)
        self.random_occlu_idx = self.random_occlu_count > 0

        self.random_occlu_idx[:] = True
        self.random_occlu_idx[:, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = False

    def _action_to_pd_targets(self, action):
        if self._res_action:
            pd_tar = self.ref_dof_pos + self._pd_action_scale * action
            pd_lower = self._dof_pos - np.pi / 2
            pd_upper = self._dof_pos + np.pi / 2
            pd_tar = torch.maximum(torch.minimum(pd_tar, pd_upper), pd_lower)

        else:
            pd_tar = self._pd_action_offset + self._pd_action_scale * action

        return pd_tar
    

    def pre_physics_step(self, actions):

        super().pre_physics_step(actions)
        self._update_cycle_count()

        if self._occl_training:
            self._update_occl_training()

        return

    def _compute_reset(self):
        time = (self.progress_buf) * self.dt + self._motion_start_times + self._motion_start_times_offset # Reset is also called after the progress_buf is updated. 

        pass_time_max = self.progress_buf >= self.max_episode_length - 1
        pass_time_motion_len = time >= self._motion_lib._motion_lengths
        
        if self.cycle_motion:
            pass_time = pass_time_max
            if pass_time_motion_len.sum() > 0:
                self._motion_start_times_offset[pass_time_motion_len] = -self.progress_buf[pass_time_motion_len] * self.dt  # such that the proegress_buf will cancel out to 0.
                self._motion_start_times[pass_time_motion_len] = self._sample_time(self._sampled_motion_ids[pass_time_motion_len])
                self._cycle_counter[pass_time_motion_len] = 60

                root_res = self._motion_lib.get_root_pos_smpl(self._sampled_motion_ids[pass_time_motion_len], self._motion_start_times[pass_time_motion_len])
                if self.cycle_motion_xp:
                    self._global_offset[pass_time_motion_len, :2] = self._humanoid_root_states[pass_time_motion_len, :2] - root_res['root_pos'][:, :2] + torch.rand(pass_time_motion_len.sum(), 2).to(self.device)  # one meter
                elif self.zero_out_far and self.zero_out_far_train:

                    max_distance = 5
                    num_cycle_motion = pass_time_motion_len.sum()
                    rand_distance = torch.sqrt(torch.rand(num_cycle_motion).to(self.device)) * max_distance
                    rand_angle = torch.rand(num_cycle_motion).to(self.device) * np.pi * 2

                    self._global_offset[pass_time_motion_len, :2] = self._humanoid_root_states[pass_time_motion_len, :2] - root_res['root_pos'][:, :2] + torch.cat([(torch.cos(rand_angle) * rand_distance)[:, None], (torch.sin(rand_angle) * rand_distance)[:, None]], dim=-1)
                else:
                    self._global_offset[pass_time_motion_len, :2] = self._humanoid_root_states[pass_time_motion_len, :2] - root_res['root_pos'][:, :2]

                time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset  # update time
                if flags.test:
                    print("cycling motion")
        else:
            pass_time = pass_time_motion_len

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, time, self._global_offset)

        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        if self.zero_out_far and self.zero_out_far_train:
            # zeros_subset = torch.norm(self._rigid_body_pos[..., 0, :] - ref_rb_pos[..., 0, :], dim=-1) > self._termination_distances[..., 0]
            # zeros_subset = torch.norm(self._rigid_body_pos[..., 0, :] - ref_rb_pos[..., 0, :], dim=-1) > 0.1
            # self.reset_buf[zeros_subset], self._terminate_buf[zeros_subset] = compute_humanoid_traj_reset(
            #     self.reset_buf[zeros_subset], self.progress_buf[zeros_subset], self._contact_forces[zeros_subset],
            #     self._contact_body_ids,  self._rigid_body_pos[zeros_subset], self.max_episode_length,  self._enable_early_termination,
            #     0.3, flags.no_collision_check)

            # self.reset_buf[~zeros_subset], self._terminate_buf[~zeros_subset] = compute_humanoid_reset(
            #     self.reset_buf[~zeros_subset], self.progress_buf[~zeros_subset], self._contact_forces[~zeros_subset],
            #     self._contact_body_ids, self._rigid_body_pos[~zeros_subset][..., self._reset_bodies_id, :], ref_rb_pos[~zeros_subset][..., self._reset_bodies_id, :],
            #     pass_time[~zeros_subset], self._enable_early_termination,
            #     self._termination_distances[..., self._reset_bodies_id], flags.no_collision_check)

            # self.reset_buf, self._terminate_buf = compute_humanoid_traj_reset(  # traj reset
            #     self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, self._rigid_body_pos, pass_time_max, self._enable_early_termination, 0.3, flags.no_collision_check)
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_im_reset(  # Humanoid reset
                self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, self._rigid_body_pos[..., self._reset_bodies_id, :], ref_rb_pos[..., self._reset_bodies_id, :], pass_time, self._enable_early_termination, self._termination_distances[..., self._reset_bodies_id],
                flags.no_collision_check, flags.im_eval and (not self.strict_eval))

        else:
            body_pos = self._rigid_body_pos[..., self._reset_bodies_id, :].clone()
            ref_body_pos = ref_rb_pos[..., self._reset_bodies_id, :].clone()

            if self._occl_training:
                ref_body_pos[self.random_occlu_idx[:, self._reset_bodies_id]] = body_pos[self.random_occlu_idx[:, self._reset_bodies_id]]

            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_im_reset(self.reset_buf, self.progress_buf, self._contact_forces, self._contact_body_ids, \
                                                                               body_pos, ref_body_pos, pass_time, self._enable_early_termination,
                                                                               self._termination_distances[..., self._reset_bodies_id], flags.no_collision_check, flags.im_eval and (not self.strict_eval))
        is_recovery = torch.logical_and(~pass_time, self._cycle_counter > 0)  # pass time should override the cycle counter.
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        
        return

    def _draw_task(self):
        self._update_marker()
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_imitation_observations(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, int, bool) -> Tensor
    # We do not use any dof in observation.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot).repeat_interleave(time_steps, 0).view(B, time_steps, J, 4))

    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis

    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * 10 * 3 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * 10 * 3 * 6

    ##### Velocities
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)

    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    obs.append(diff_local_vel.view(B, -1))  # 3 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # 3 * 3

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_imitation_observations_v2(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, dof_pos, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, ref_dof_pos, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding dof
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot).repeat_interleave(time_steps, 0).view(B, time_steps, J, 4))

    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis

    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * 10 * 3 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * 10 * 3 * 6

    ##### Velocities
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)

    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    obs.append(diff_local_vel.view(B, -1))  # 3 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # 3 * 3

    ##### Dof_pos diff
    diff_dof_pos = ref_dof_pos.view(B, time_steps, -1) - dof_pos.view(B, time_steps, -1)
    obs.append(diff_dof_pos.view(B, -1))  # 23 * 3

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_imitation_observations_v3(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, int, bool) -> Tensor
    # No velocities
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * 10 * 3 * 3

    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot).repeat_interleave(time_steps, 0).view(B, time_steps, J, 4))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * 10 * 3 * 6

    obs = torch.cat(obs, dim=-1)

    return obs


@torch.jit.script
def compute_imitation_observations_v6(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * 24 * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs


@torch.jit.script
def compute_imitation_observations_v7(root_pos, root_rot, body_pos, body_vel, ref_body_pos, ref_body_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, int, bool) -> Tensor
    # No rotation information. Leave IK for RL.
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)

    ##### Body position differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    ##### Linear Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))

    ##### body pos + Dof_pos 
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * 10 * 3 * 3
    obs.append(diff_local_vel.view(B, time_steps, -1))  # 3 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # 2

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs

@torch.jit.script
def compute_imitation_observations_v8(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1))
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1))

    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3)[:, 0:1] - body_pos.view(B, 1, J, 3)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4)[:, 0:1], torch_utils.quat_conjugate(body_rot).view(B, 1, J, 4))

    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis

    ##### Body position differences
    obs.append(diff_local_body_pos_flat.view(B, -1))  # 1 * 10 * J * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, -1))  #  1 * 10 * J * 6

    ##### Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3)[:, 0:1] - body_vel.view(B, 1, J, 3)
    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3)[:, 0:1] - body_ang_vel.view(B, 1, J, 3)

    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    obs.append(diff_local_vel.view(B, -1))  # 24 * 3
    obs.append(diff_local_ang_vel.view(B, -1))  # 24 * 3

    ##### body pos + Dof_pos This part will have proper futuers.
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    local_ref_body_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))
    local_ref_body_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_ang_vel.view(-1, 3))

    # make some changes to how futures are appended.
    if time_steps > 1:
        local_ref_body_pos = local_ref_body_pos.view(B, time_steps, -1)
        local_ref_body_rot = local_ref_body_rot.view(B, time_steps, -1)

        obs.append(local_ref_body_pos[:, 0].view(B, -1))  # first append the current ones
        obs.append(local_ref_body_rot[:, 0].view(B, -1))
        obs.append(local_ref_body_vel[:, 0].view(B, -1))
        obs.append(local_ref_body_ang_vel[:, 0].view(B, -1))


        obs.append(local_ref_body_pos[:, 1:].reshape(B, -1))  # then append the future ones
        obs.append(local_ref_body_rot[:, 1:].reshape(B, -1))
        obs.append(local_ref_body_vel[:, 1:].view(B, -1))
        obs.append(local_ref_body_ang_vel[:, 1:].view(B, -1))
    else:
        obs.append(local_ref_body_pos.view(B, -1))  # 24 * timestep * 3
        obs.append(local_ref_body_rot.view(B, -1))  # 24 * timestep * 6
        obs.append(local_ref_body_vel.view(B, -1))  # 24 * timestep * 3
        obs.append(local_ref_body_ang_vel.view(B, -1))  # 24 * timestep * 3
    # obs.append(ref_dof_pos.view(B, -1)) # 23 * 3

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_imitation_observations_v9(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_root_vel, ref_body_root_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    
    ##### linear and angular  Velocity differences
    heading_inv_rot_expand_root = heading_inv_rot.unsqueeze(-1).repeat_interleave(time_steps, 0)
    root_vel, root_ang_vel = body_vel[:, 0], body_ang_vel[:, 0]
    diff_global_root_vel = ref_root_vel.view(B, time_steps, 3) - root_vel.view(B, 1, 3)
    diff_local_root_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand_root.view(-1, 4), diff_global_root_vel.view(-1, 3))


    diff_global_root_ang_vel = ref_body_root_ang_vel.view(B, time_steps, 3) - root_ang_vel.view(B, 1, 3)
    diff_local_root_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand_root.view(-1, 4), diff_global_root_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * 10 * 3 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * 10 * 3 * 6
    obs.append(diff_local_root_vel.view(B, time_steps, -1))  # 3 * 3
    obs.append(diff_local_root_ang_vel.view(B, time_steps, -1))  # 3 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # 24 * timestep * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # 24 * timestep * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs


@torch.jit.script
def compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, rwd_specs):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Dict[str, float]) -> Tuple[Tensor, Tensor]
    k_pos, k_rot, k_vel, k_ang_vel = rwd_specs["k_pos"], rwd_specs["k_rot"], rwd_specs["k_vel"], rwd_specs["k_ang_vel"]
    w_pos, w_rot, w_vel, w_ang_vel = rwd_specs["w_pos"], rwd_specs["w_rot"], rwd_specs["w_vel"], rwd_specs["w_ang_vel"]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body rotation reward
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot, torch_utils.quat_conjugate(body_rot))
    diff_global_body_angle = torch_utils.quat_to_angle_axis(diff_global_body_rot)[0]
    diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # body angular velocity reward
    diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)

    reward = w_pos * r_body_pos + w_rot * r_body_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)
    # import ipdb
    # ipdb.set_trace()
    return reward, reward_raw


@torch.jit.script
def compute_point_goal_reward(prev_dist, curr_dist):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
    reward = torch.clamp(prev_dist - curr_dist, max=1 / 3) * 9

    return reward, reward


@torch.jit.script
def compute_location_reward(root_pos, tar_pos):
    # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
    pos_err_scale = 1.0

    pos_diff = tar_pos[..., 0:2] - root_pos[..., 0:2]

    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward

    return reward, reward


@torch.jit.script
def compute_humanoid_im_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, ref_body_pos, pass_time, enable_early_termination, termination_distance, disableCollision, use_mean):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, Tensor, bool, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)
    if (enable_early_termination):
        if use_mean:
            has_fallen = torch.any(torch.norm(rigid_body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance[0], dim=-1)  # using average, same as UHC"s termination condition
        else:
            has_fallen = torch.any(torch.norm(rigid_body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1)  # using max
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        if disableCollision:
            has_fallen[:] = False
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

        # if (contact_buf.abs().sum(dim=-1)[0] > 0).sum() > 2:
        #     np.set_printoptions(precision=4, suppress=1)
        #     print(contact_buf.numpy(), contact_buf.abs().sum(dim=-1)[0].nonzero().squeeze())

        # if terminated.sum() > 0:
        #     import ipdb; ipdb.set_trace()
        #     print("Fallen")

    reset = torch.where(pass_time, torch.ones_like(reset_buf), terminated)
    # import ipdb
    # ipdb.set_trace()

    return reset, terminated


@torch.jit.script
def compute_location_observations(root_pos, root_rot, target_pos, upright):
    # type: (Tensor, Tensor, Tensor, bool) -> Tensor

    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)

    diff_global_body_pos = target_pos - root_pos
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot, diff_global_body_pos.view(-1, 3))
    max_distance = 7.5
    distances = torch.norm(diff_local_body_pos_flat, dim=-1)
    smallers = distances < max_distance  # 2.5 seconds, 5 time steps,
    diff_locations = torch.zeros((smallers.shape[0], 5, 3)).to(diff_local_body_pos_flat)
    diff_locations[smallers] = (diff_local_body_pos_flat[smallers, None] * torch.linspace(0.2, 1, 5)[None, :, None].repeat(smallers.sum(), 1, 1).to(diff_local_body_pos_flat))  # 5 time stpes, 2 seconds
    modified_locals = diff_local_body_pos_flat[~smallers] * distances[~smallers, None] / max_distance
    diff_locations[~smallers] = modified_locals[:, None] * torch.linspace(0.2, 1, 5)[None, :, None].repeat((~smallers).sum(), 1, 1).to(diff_local_body_pos_flat)

    local_traj_pos = diff_locations[..., 0:2]

    obs = torch.reshape(local_traj_pos, (local_traj_pos.shape[0], -1))
    return obs


@torch.jit.script
def compute_humanoid_traj_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, pass_time, enable_early_termination, termination_heights, disableCollision):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        ## torch.sum to disable self-collision.
        # force_threshold = 200
        force_threshold = 50
        body_contact_force = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        has_contacted_fall = body_contact_force
        has_contacted_fall *= (progress_buf > 1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_failed = torch.logical_and(has_contacted_fall, fall_height)

        if disableCollision:
            has_failed[:] = False

        ############################## Debug ##############################
        # if torch.sum(has_fallen) > 0:
        #     import ipdb; ipdb.set_trace()
        #     print("???")
        # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'])
        # print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
        ############################## Debug ##############################

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(pass_time, torch.ones_like(reset_buf), terminated)

    return reset, terminated

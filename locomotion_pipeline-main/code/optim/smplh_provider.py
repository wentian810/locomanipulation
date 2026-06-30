import os, sys
import glob
import time

pwd = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.abspath(os.path.join(pwd, '..'))
if not code_root in sys.path:
    sys.path.insert(0, code_root)

import numpy as np
import torch
import smplx
import trimesh
from tqdm import tqdm

from utils.log_utils import MyLogger
from utils.statics import SequenceStatistics
from core.penetration import Penetration
from utils.vis3d_utils import Visualization3d
from core.filter import PoseFilter

# root_path = '/home/ubuntu/Synadata_dev/zyh/datapipe_v3'
# root_path = '/workspace/projects_dataset/data_deliver/smpl2qiaojie'

class Smplh_sequence:
    def __init__(
        self, 
        param_path:str, 
        logger: MyLogger, 
        device:str='cuda', 
    ):
        self.param_path = param_path
        self.logger = logger
        
        self.device = device
        if 'cuda' in device:
            assert torch.cuda.is_available()

        self.penetraction_threshold = 0.1
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.load_smplh_seg()

        if param_path.endswith('npz'):
            self.load_smpl_npz()
        else:
            self.load_smpl_pkl()
    
    def load_smplh_seg(self,):
        import json

        with open('assets/smplh-seg/arm_leg_seg.json', 'r') as f:
            seg_data = json.load(f)
        
        selected_indices = []
        for part_name, vertex_indices in seg_data.items():
            # print(f"{part_name}: {len(vertex_indices)} vertices")
            selected_indices.append(
                np.array(vertex_indices, dtype=np.int64)
            )

        self.query_point_mask = selected_indices

        with open('assets/smplh-seg/arm_leg_seg_faces.json', 'r') as f:
            seg_faces_data = json.load(f)
        selected_face_indices = []
        for part_name, face_indices in seg_faces_data.items():
            selected_face_indices.append(
                np.array(face_indices['face_indices'], dtype=np.int64)
            )
        
        self.query_face_mask = selected_face_indices

        # 反选 faces
        whole_faces = set(range(13776))
        converted_face_mask = []
        for face_indices in selected_face_indices:
            converted_face_mask.append(np.array(
                list(whole_faces - set(face_indices)), dtype=np.int64
            ))
        self.converted_face_mask = converted_face_mask

    def load_smpl_pkl(self):
        smplh_data = np.load(self.param_path, allow_pickle=True)
        self.body_pose   = torch.tensor(smplh_data['body_pose']).float() # (T, 63)
        self.betas       = torch.tensor(smplh_data['betas']).float() # (T, 10)
        self.root_orient = torch.tensor(smplh_data['global_orient']).float() # (T, 3)
        self.trans       = torch.tensor(smplh_data['transl']).float() # (T, 3)

        # Build poses (T, 24, 3) compatible with npz export format
        T = self.body_pose.shape[0]
        poses = torch.zeros(T, 24, 3)
        poses[:, 0, :] = self.root_orient
        poses[:, 1:22, :] = self.body_pose.reshape(T, 21, 3)
        self.poses = poses

        self.trans_original = self.trans.clone()
        self.mocap_frame_rate = float(smplh_data.get('fps', 30))
        self.gender      = 'neutral'

        self.num_frames = smplh_data["body_pose"].shape[0]

    def load_smpl_npz(self):
        '''
        npz keys:
            - poses (T, 24, 3)
            - trans (T, 3)
            - gender ()
            - mocap_frame_rate ()
            - betas (16,)
            - root_orient (T, 3)
            - pose_body (T, 63)
            - trans_original (T, 3)
        '''
        smplh_data = np.load(self.param_path, allow_pickle=True)

        self.body_pose   = torch.tensor(smplh_data['pose_body']).float()   # (T, 63)
        self.betas = torch.tensor(smplh_data['betas']).float() # (16)
        self.root_orient = torch.tensor(smplh_data['root_orient']).float() # (T, 3)
        self.trans = torch.tensor(smplh_data['trans']).float() # (T, 3)

        self.poses = torch.tensor(smplh_data['poses']).float() # (T, 24, 3)
        self.trans_original = torch.tensor(smplh_data['trans_original']).float() \
            if 'trans_original' in smplh_data else \
            torch.tensor(smplh_data['trans']).float() # (T, 3)
        self.gender = str(smplh_data['gender']) # e.g. 'neutral'
        self.mocap_frame_rate = float(smplh_data['mocap_frame_rate']) # a value
        
        self.num_frames = smplh_data["pose_body"].shape[0]
        # self.param2shape()

    def param2shape(self,):
        betas = self.betas
        if betas.ndim == 1:
            betas = betas.view(1, -1).repeat([self.num_frames, 1])
        
        num_betas = betas.shape[1]
        self.body_model = smplx.SMPLH(
            model_path='assets/smplh',
            gender=self.gender, 
            num_betas=num_betas, 
            use_pca=False, 
        )
        
        self.smplh_output = self.body_model(
            betas=betas, # (N, 16)
            global_orient=self.root_orient, # (N, 3)
            body_pose=self.body_pose, # (N, 63)
            transl=self.trans, # (N, 3)
            left_hand_pose=torch.zeros(self.num_frames, 45).float(),
            right_hand_pose=torch.zeros(self.num_frames, 45).float(),
            return_full_pose=True,
        )

    def check_gravity_axis(self, gravity_axis:str):
        '''
        Check if the gravity axis is valid.
        should be in the format of (+/-)(x/y/z), e.g. +y, -z, etc.
        '''
        assert len(gravity_axis) == 2
        assert gravity_axis[0] in ['+', '-']
        assert gravity_axis[1].lower() in ['x', 'y', 'z']

    def export_shapes(
        self, 
        export_path: str, 
        color_pc: bool = False,
        gravity_axis: str = 'y-',
        gravity_info: dict = None,
        frame_indices: list = None,
    ):
        faces = self.body_model.faces

        if not os.path.exists(export_path):
            os.makedirs(export_path)

        ext = 'ply' if color_pc else 'obj'
        frame_indices = frame_indices if frame_indices is not None else range(self.num_frames)

        if color_pc:
            gravity_axis = gravity_axis[::-1]
            self.check_gravity_axis(gravity_axis)
            gravity_axis_idx = self.axis_map[gravity_axis[1].lower()]

            if gravity_info is not None:
                in_support_indices = gravity_info['in_support_indices']
            else:
                in_support_indices = set()

            for i in tqdm(
                frame_indices,  
                total=len(frame_indices), 
            ):
                vertices = self.smplh_output.vertices[i] # (V, 3), tensor
                vertex_colors = np.ones_like(vertices.detach().cpu().numpy()) # (V, 3)
                if i in in_support_indices:
                    vertex_colors *= np.array([0, 1, 0]) # 支撑帧的点标记为绿色
                # 对于地面附近0.1m的点，标记为红色
                contact_mask1 = vertices[:, gravity_axis_idx] < self.penetraction_threshold # (V,)
                contact_mask2 = vertices[:, gravity_axis_idx] > -self.penetraction_threshold # (V,)
                contact_mask = contact_mask1 & contact_mask2
                vertex_colors[contact_mask] = np.array([1, 0, 0])
                pointcloud = trimesh.PointCloud(vertices=vertices.detach().cpu().numpy(), colors=vertex_colors)
                pointcloud.export(f'{export_path}/frame_{i:04d}.{ext}')
        else:
            for i in tqdm(
                frame_indices,  
                total=len(frame_indices), 
            ):
                vertices = self.smplh_output.vertices[i] # (V, 3), tensor
                mesh = trimesh.Trimesh(vertices=vertices.detach().cpu().numpy(), faces=faces)
                mesh.export(f'{export_path}/frame_{i:04d}.{ext}')

    def export_parameters(
        self,
        save_path:str = None
    ):
        '''

        '''
        # Use save_path extension to determine output format, not param_path
        path_for_ext = save_path if save_path is not None else self.param_path
        ext_seg_index = path_for_ext.rfind('.')
        ext = path_for_ext[ext_seg_index+1:]
        if save_path is None:
            path_no_ext = self.param_path[:self.param_path.rfind('.')]
            save_path = f'{path_no_ext}_optim.{ext}'

        if ext == 'pkl':
            import pickle
            save_dict = {
                'body_pose': self.body_pose.numpy(), 
                'betas': self.betas.numpy(),
                'global_orient': self.root_orient.numpy(),
                'transl': self.trans.numpy(),
            }
            with open(save_path, 'wb') as f:
                pickle.dump(save_dict, f)
                
        elif ext == 'npz':
            save_dict = {
                'poses': self.poses.numpy(), 
                'trans': self.trans.numpy(), 
                'betas': self.betas.numpy(), 
                'root_orient': self.root_orient.numpy(),
                'pose_body': self.body_pose.numpy(), 
                'trans_original': self.trans_original.numpy(), 
                'gender': self.gender,
                'mocap_frame_rate': self.mocap_frame_rate,
            }
            np.savez(
                save_path, 
                **save_dict, 
            )

    def optimize_height(
        self, 
        gravity_axis:str, 
        height_scale:float = 1.0, 
    ):
        '''
        Args:
            - gravity: (X/Y/Z)(+/-)

        e.g. gravity_axis == 'z+'
        '''
        gravity_axis = gravity_axis[::-1]
        assert len(gravity_axis)==2
        assert gravity_axis[0] in ['+', '-']
        assert gravity_axis[1].lower() in ['x', 'y', 'z']
        
        axis_idx = self.axis_map[gravity_axis[1].lower()]
        gravity_sign = 1.0 if gravity_axis[0] == '+' else -1.0

        # Use the signed axis value so "larger" always means further along gravity direction.

        # 先在指定坐标轴上做一个缩放，从而让角色整体更接近地面，减少离谱的情况对后续统计的影响
        if not np.isclose(height_scale, 1.0):
            self.trans[:, axis_idx] *= height_scale

        self.param2shape()
        vertices = self.smplh_output.vertices.detach().cpu().numpy() # (T, V, 3)
        signed_vertices = gravity_sign * vertices[:, :, axis_idx]    # (T, V)

        # Per-frame contact value: use FOOT JOINTS only (not all vertices).
        # Joints 7=L_ankle, 8=R_ankle, 10=L_foot, 11=R_foot in SMPL.
        # Using all vertices can pick up hands when bending forward (hand > foot),
        # which shifts the ground plane to hand height and makes feet float.
        foot_joints = self.smplh_output.joints[:, [7, 8, 10, 11], :]        # (T, 4, 3)
        foot_joints = foot_joints.detach().cpu().numpy()                     # → numpy
        foot_signed = gravity_sign * foot_joints[:, :, axis_idx]             # (T, 4)
        frame_contact = np.max(foot_signed, axis=1)                          # (T,)

        q1, q3 = np.quantile(frame_contact, [0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        inliers = frame_contact[(frame_contact >= lower) & (frame_contact <= upper)]

        # compute ratio of outliers before optimization and print it
        out_of_range_mask_before = (frame_contact < lower) | (frame_contact > upper)
        out_of_range_ratio_before = float(np.mean(out_of_range_mask_before))
        self.logger.info(f'Before optimization: Out of range ratio: {out_of_range_ratio_before}')

        robust_offset = np.median(inliers) if inliers.size > 0 else np.median(frame_contact)
        adjusted_frame_contact = frame_contact - robust_offset
        # adjusted_lower = lower - robust_offset
        # adjusted_upper = upper - robust_offset
        adjusted_lower = -0.1
        adjusted_upper = 0.1
        self.logger.info(f'Adjusted contact range: [{adjusted_lower}, {adjusted_upper}]')
        out_of_range_mask = (adjusted_frame_contact < adjusted_lower) | (adjusted_frame_contact > adjusted_upper)
        out_of_range_count = np.sum(out_of_range_mask)
        total_frames = self.num_frames
        out_of_range_ratio = float(np.mean(out_of_range_mask))
        self.logger.info(f'Out of range ratio: {out_of_range_ratio}')
        delta_axis = -gravity_sign * robust_offset

        self.trans[:, axis_idx] += float(delta_axis)
        self.param2shape()

        return {
            'gravity_axis': gravity_axis,
            'axis_index': axis_idx,
            'offset': float(robust_offset),
            'delta_axis': float(delta_axis),
            'adjusted_contact_range': (float(adjusted_lower), float(adjusted_upper)),
            'adjusted_contact_out_of_range_ratio': out_of_range_ratio,
            'out_of_range_count': out_of_range_count,
            'out_of_range_ratio_before': out_of_range_ratio_before,
            'total_frames': total_frames,
        }

    def check_gravity_center(
        self, 
        gravity_axis:str, 
        scale:float = 1.0,
    ):
        # 计算重心位置
        gravity_axis = gravity_axis[::-1]
        self.check_gravity_axis(gravity_axis)
        assert scale > 0, 'scale should be positive.'

        # 计算重心位置
        # e.g. 比如重力轴是 -y, 那么投影到 xz 平面上，计算重心在 xz 平面上的位置分布
        plane_axes = [i for i in range(3) if i != self.axis_map[gravity_axis[1].lower()]]
        vertices = self.smplh_output.vertices.detach().cpu().numpy() # (T, V, 3)
        projected_vertices = vertices[:, :, plane_axes] # (T, V, 2)
        center_of_mass = np.mean(projected_vertices, axis=1) # (T, 2)

        # 得到重心位置后，计算肢体与地面的接触情况，检查重心是否落在支撑范围内
        # 这里假设 浮空/穿地 在 0.1m 以内的作为接触，计算接触点的 obb 作为支撑范围，检查重心位置与支撑范围的关系
        gravity_axis_idx = self.axis_map[gravity_axis[1].lower()]
        signed_vertices = vertices[:, :, gravity_axis_idx] # (T, V)
        contact_mask = signed_vertices < 0.1  # (T, V)
        # 对每一帧都计算接触点, 用凸包可能比较麻烦，算 oriented-bounding-box 会简单一些

        in_support_indices = set()
        in_support_num = 0

        pass_frame_indices = []
        for t_idx in range(self.num_frames):
            contact_points = vertices[t_idx][contact_mask[t_idx]] # (N_contact, 3)
            if contact_points.shape[0] < 3:
                # self.logger.info(f'Frame {t_idx}: Not enough contact points to compute support area.')
                pass_frame_indices.append(t_idx)
                continue
            contact_points_2d = contact_points[:, plane_axes] # (N_contact, 2)
            # 计算 oriented bounding box， 用 pca 即可
            cov = np.cov(contact_points_2d, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, order]
            projected_center = center_of_mass[t_idx] @ eigenvectors # (2,)
            obb_half_extent = 3 * np.sqrt(eigenvalues[order])
            if np.all(np.abs(projected_center) <= scale * obb_half_extent):
                in_support_indices.add(t_idx)
                in_support_num += 1
            
        if len(pass_frame_indices) > 0:
            self.logger.info(f'insufficient contact points in frames: {pass_frame_indices}, \ntotal {len(pass_frame_indices)} frames.')

        return {
            'gravity_axis': gravity_axis,
            'support_scale': float(scale),
            'support_ratio': in_support_num / self.num_frames,
            'in_support_indices': in_support_indices,
            'in_support_num': in_support_num,
        }

    def check_smooth(
        self, 
        max_velocity: float = 1.0,
    ):
        '''
        计算 global orientation & translation 的速度，检查是否平滑
        '''

    def check_penetration(
        self,
        depth_threshold: float = 0.1,
        sample_ratio: int = 3,
    ):
        '''
        自穿模程度 — 使用纯 numpy KDTree + 顶点法向量方法（不需 CUDA）
        '''
        import numpy as np

        if not hasattr(self, 'smplh_output'):
            self.param2shape()

        vertices = self.smplh_output.vertices.detach().cpu().numpy()  # (T, V, 3)
        faces = self.body_model.faces                                # (F, 3)

        # 构建 query_mask: 使用 load_smplh_seg() 加载的肢体分段顶点
        num_vertices = vertices.shape[1]
        query_mask = np.zeros((len(self.query_point_mask), num_vertices), dtype=bool)
        for batch_idx, point_indices in enumerate(self.query_point_mask):
            query_mask[batch_idx, point_indices] = True

        detector = Penetration()
        result = detector.check_self_penetration(
            mesh_vertices=vertices,
            mesh_faces=faces,
            query_mask=query_mask,
            depth_threshold=depth_threshold,
        )
        return result

    def check_speed(
        self, 
        choices: dict
    ):
        '''
        compute linear/angular velocity/acceleration, check if they are within reasonable range
        '''
        result = {}

        pf = PoseFilter(self.mocap_frame_rate)
        if 'linear' in choices.keys():
            calc_acceleration = ('acceleration' in choices['linear'])
            linear_speed = pf.calculate_linear_speed(
                self.trans.numpy(),
                calc_acceleration=calc_acceleration, 
                smooth=True, 
            )
            result['linear'] = {}

            for property_name, upperbound in choices['linear'].items():
                speed = linear_speed[property_name] # (T-1, 3) or (T-2, 3)
                outlier_mask = np.any(speed > upperbound, axis=-1) # (T-1,) or (T-2,)
                result['linear'][property_name] = outlier_mask.sum()

        if 'angular' in choices.keys():
            raise NotImplementedError('Angular speed check is not implemented yet.')
            # angular_speed = pf.calculate_angular_speed(self.rot.numpy())

        return result

def update_bash(seq_name):
    pwd = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../..'
    ))

    with open(f'{pwd}/run_zip.sh', 'r') as f:
        lines = f.readlines()

    with open(f'{pwd}/run_zip.sh', 'w') as f:
        for line in lines:
            if line.startswith('seq_name='):
                f.write(f'seq_name={seq_name}\n')
            else:
                f.write(line)

def batch_process(
    path_pattern: str,
    logger: MyLogger,
    seqID_index: int = 0, # 0 表示文件名即为 sequence ID, 1 表示父目录名为 sequence ID， 以此类推
    gravity_axis: str = 'y-',
    height_scale: float = 1.0,
    run_optim: bool = True,
    run_detect_penetration: bool = False,
    check_gravity: bool = True,
    path_export_shapes: str = None,
    path_parameters: str = None,
    **kwargs,
):
    input_paths = glob.glob(path_pattern)
    height_statics = SequenceStatistics()
    gravity_statics = SequenceStatistics()

    for input_path in tqdm(input_paths):
        fname_ext = os.path.basename(input_path)
        dot_index = fname_ext.rfind('.')
        fname, ext = fname_ext[:dot_index], fname_ext[dot_index:]
        
        tmp_path = input_path
        for _ in range(seqID_index):
            tmp_path = os.path.dirname(tmp_path)
        seq_id = os.path.basename(tmp_path)

        if ext not in ['.npz', '.pkl']:
            logger.info(f'Skipping unsupported file: {input_path}')
            continue

        smplh_seq = Smplh_sequence(input_path, logger=logger)
        if path_export_shapes is not None and 'init_shape_folder_name' in kwargs:
            os.makedirs(path_export_shapes, exist_ok=True)
            print(f'Exporting initial shapes for sequence {seq_id}...')
            if not hasattr(smplh_seq, 'smplh_output'):
                smplh_seq.param2shape()
            folder_name = kwargs['init_shape_folder_name']
            smplh_seq.export_shapes(
                f'{path_export_shapes}/{folder_name}/{seq_id}', 
                color_pc=kwargs.get('color_pc', False),
                gravity_axis=gravity_axis,
                frame_indices=kwargs.get('init_shape_frame_indices', None),
            )
        if run_optim:
            optim_info = smplh_seq.optimize_height(gravity_axis, height_scale)
            logger.recursive_log(
                {seq_id: optim_info}, 
                prefix='Optim Info - '
            )
            height_statics.add(
                seq_id, 
                optim_info['total_frames'], 
                optim_info['total_frames'] - optim_info['out_of_range_count']
            )
        if check_gravity:
            gravity_info = smplh_seq.check_gravity_center(gravity_axis, scale=1.1)
            logger.recursive_log(
                {seq_id: gravity_info}, 
                prefix='Gravity Info - ',
                traversal_iter=False,
            )
            gravity_statics.add(
                seq_id, 
                smplh_seq.num_frames, 
                gravity_info['in_support_num']
            )
        if run_optim and path_export_shapes is not None:
            os.makedirs(path_export_shapes, exist_ok=True)
            if not check_gravity:
                gravity_info = None
            smplh_seq.export_shapes(
                f'{path_export_shapes}/optim/{seq_id}', 
                color_pc=kwargs.get('color_pc', False),
                gravity_axis=gravity_axis, 
                gravity_info=gravity_info,
                frame_indices=kwargs.get('optim_shape_frame_indices', None),
            )
        if run_optim and path_parameters is not None:
            os.makedirs(path_parameters, exist_ok=True)
            smplh_seq.export_parameters(f'{path_parameters}/{fname}{ext}')
        if run_detect_penetration:
            print(f'Checking penetration for sequence {seq_id}...')
            export_folder = f'{path_export_shapes}/penetration/{seq_id}' if path_export_shapes is not None else None

            sample_ratio = 3
            if not hasattr(smplh_seq, 'smplh_output'):
                smplh_seq.param2shape()
            detection_result = smplh_seq.check_penetration(
                depth_threshold=0.1, 
                sample_ratio=sample_ratio
            )

            # visualization
            if export_folder is not None:
                logger.message('Exporting penetration visualization...')
                os.makedirs(export_folder, exist_ok=True)
                for i in range(0, smplh_seq.num_frames, sample_ratio):
                    vertices = smplh_seq.smplh_output.vertices[i].detach().cpu().numpy()
                    # color_depth = detection_result['penetration_depth'][i // sample_ratio] # (V)
                    color_depth = detection_result['penetration_mask'][i // sample_ratio] * 1.0 # (V)

                    Visualization3d.export_single_mesh(
                        vertices=vertices, 
                        faces=smplh_seq.body_model.faces,
                        point_color=color_depth, 
                        export_path=f'{export_folder}/frame_{i:04d}.obj', 
                    )
        # break # for debug, process only one file

    out = {}
    if run_optim:
        out['height'] = height_statics
    if check_gravity:
        out['gravity'] = gravity_statics

    return out

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--seqID_index', type=int, required=True)
    parser.add_argument('--file_pattern', type=str, required=True)
    parser.add_argument('--gravity_axis', type=str, required=True)
    parser.add_argument('--log_name', type=str, default='smplh_provider.log')
    parser.add_argument('--run_optim', type=int, required=True)
    parser.add_argument('--run_detect_penetration', type=int, default=0)
    parser.add_argument('--check_gravity', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    target_dir = args.target_dir
    save_dir = args.save_dir
    logger = MyLogger(
                name=args.log_name, 
                log_dir='log', 
                log_filename=args.log_name, 
            )
    out = batch_process(
        path_pattern=f'{target_dir}/{args.file_pattern}', 
        logger=logger, 
        seqID_index=args.seqID_index, 
        gravity_axis=args.gravity_axis, 
        height_scale=1, 
        run_optim=args.run_optim, 
        run_detect_penetration=args.run_detect_penetration, 
        check_gravity=args.check_gravity, 
        path_export_shapes=save_dir, 
        path_parameters=None, 
        **{
            'color_pc': True,
            # 'init_shape_frame_indices': None,
            # 'optim_shape_frame_indices': None,
            'init_shape_folder_name': 'gvhmr_motor',
        }
    )

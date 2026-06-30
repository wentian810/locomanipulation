import os, sys

pwd = os.path.dirname(os.path.abspath(__file__))
code_root = os.path.abspath(os.path.join(pwd, '..'))
if not code_root in sys.path:
    sys.path.insert(0, code_root)

import glob
import numpy as np
from tqdm import tqdm
from typing import Optional, Literal

from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter

from utils.vis_utils import visualize2d

class PoseMath:
    @staticmethod
    def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
        '''
        Shape:
        - axis_angle: (..., 3)
        - return: (..., 4)
        '''
        angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
        half = 0.5 * angle
        axis = np.divide(axis_angle, np.clip(angle, 1e-12, None))
        xyz = axis * np.sin(half)
        w = np.cos(half)
        quat = np.concatenate([xyz, w], axis=-1)
        return PoseMath.normalize_quaternion(quat)

    @staticmethod
    def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
        '''
        Shape:
        - quat: (..., 4)
        - return: (..., 4)
        '''
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        norm = np.clip(norm, 1e-12, None)
        return quat / norm

    @staticmethod
    def quat_conj(q: np.ndarray) -> np.ndarray:
        '''
        Shape:
        - q: (..., 4)
        - return: (..., 4)
        '''
        out = q.copy()
        out[..., :3] *= -1.0
        return out

    @staticmethod
    def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        '''
        Shape:
        - a: (..., 4)
        - b: (..., 4)
        - return: (..., 4)
        '''
        ax, ay, az, aw = np.moveaxis(a, -1, 0)
        bx, by, bz, bw = np.moveaxis(b, -1, 0)
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        w = aw * bw - ax * bx - ay * by - az * bz
        return np.stack([x, y, z, w], axis=-1)
    
    @staticmethod
    def ensure_quat_continuity(quat: np.ndarray) -> np.ndarray:
        '''
        Shape:
        - quat: (N, J, 4)
        - return: (N, J, 4)
        '''
        q = quat.copy()
        for i in range(1, q.shape[0]):
            dot = np.sum(q[i] * q[i - 1], axis=-1, keepdims=True)
            flip = dot < 0
            q[i] = np.where(flip, -q[i], q[i])
        return q

class Filter:
    def __init__(self):
        '''
        离群检测 + 插值修复
        - 离群检测：基于统计学方法（如Z-score）或机器学方法（如孤立森林）检测离群帧
        - 插值修复：离群帧连续超过一定数量，直接丢弃整个序列；离群帧数量较少，进行插值修复
        '''
        pass

    def detect_outliers(
        self, 
        data: np.ndarray, 
        mode: dict, 
    ) -> np.ndarray:
        '''
        检测离群帧
        Args:
            data: 输入数据，形状为 (T, D)，其中 T 是帧数，D 是每帧的数据维度
            mode: {
                'threshold': 阈值法, {
                    'upper_threshold': 数值,
                    'lower_threshold': 数值,
                }
                'std': 标准差法, {
                    'std_multiplier': 数值,  # 离群帧定义为超过均值±std_multiplier*标准差的帧
                }
                'MAD': 中位数绝对偏差法, {
                    'mad_multiplier': 数值,  # 离群帧定义为超过中位数±mad_multiplier*MAD的帧
                }
                'ml': 机器学习法, {
                    'model_type': 模型名称, 
                    'model_params': 模型参数字典,
                }
            }
        Returns:
            outliers: 布尔数组，形状为 (T,)，True 表示对应帧是离群帧
        '''
        
        if data.ndim == 1:
            data = data[:, np.newaxis]

        outliers_dict = {}
        if 'threshold' in mode:
            upper_threshold = mode['threshold']['upper_threshold']
            lower_threshold = mode['threshold']['lower_threshold']
            outliers_dict['threshold'] = {
                'mask': (data > upper_threshold) | (data < lower_threshold), 
                'upper_bound': upper_threshold,
                'lower_bound': lower_threshold,
            }

        if 'std' in mode:
            std_multiplier = mode['std']['std_multiplier']
            # Calculate mean and standard deviation
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Create outlier mask
            upper_bound = mean + std_multiplier * std
            lower_bound = mean - std_multiplier * std
            outliers_dict['std'] = {
                'mask': (np.abs(data - mean) > std_multiplier * std).any(axis=1),
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
            }

        if 'MAD' in mode:
            mad_multiplier = mode['MAD']['mad_multiplier']
            # Calculate median and MAD
            median = np.median(data, axis=0) # shape=(D, )
            mad = np.median(np.abs(data - median), axis=0) # shape=(D, )
            robust_sigma = 1.4826 * (mad + 1e-6)  # 加上一个小常数以避免除零
            # Create outlier mask
            print(f'robust_sigma.shape = {robust_sigma.shape} median.shape={median.shape} mad.shape={mad.shape}')
            upper_bound = median + mad_multiplier * robust_sigma
            lower_bound = median - mad_multiplier * robust_sigma
            outliers_dict['MAD'] = {
                'mask': (
                    (data > upper_bound) | (data < lower_bound)
                ).any(axis=1), 
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
            }
            
        if 'ml' in mode:
            model_type = mode['ml']['model_type']
            if model_type == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                model_params = mode['ml'].get('model_params', {'contamination': 'auto'})
                model = IsolationForest(**model_params)  # 假设离群帧占比为5%
                model.fit(data)
                outliers_dict['ml'] = {
                    'mask': model.predict(data) == -1,  # -1表示离群点
                }
            else:
                '''
                deepseek 的建议是针对输入的情况选择模型，
                快速上手、数据量较大	孤立森林 + PyOD
                需要检测局部密度异常    LOF
                数据有明显时序依赖	    自编码器 或 GRU-AE-GMM
                需要可解释性	       PCA / 椭圆包络
                不确定选哪个	       PyOD 中同时跑多个模型，对比结果后选择
                '''
                raise NotImplementedError(f"Unsupported model type: {model_type}, wait to be implemented")

        return outliers_dict

class PoseFilter(Filter):
    def __init__(
        self, 
        sequence_fps: float = 30.0,
    ):
        '''
        1. 计算 global orientation & translation 的速度，检查是否平滑
        2. 检查姿态数据中的异常值
            - 检测方法：基于统计学方法（如Z-score）或机器学习方法（如孤立森林）检测离群帧
        3. 对姿态数据进行插值处理
            - 离群帧连续超过一定数量，直接丢弃整个序列
            - 离群帧数量较少，进行插值修复
        4. 数值与可视化检测
        5. 统计分析与导出
        '''
        super().__init__()
        self.sequence_fps = sequence_fps
        pass

    # 输入检查
    def check_input(
        self, 
        rotations: np.ndarray, 
        translations: np.ndarray, 
    ) -> bool:
        '''
        检查输入数据的有效性和一致性
        Args:
            rotations: (T, 3)的旋转数据
            translations: (T, 3)的平移数据
            timestamp这里不检查，在 self 里面设置帧率就行
        Returns:
            bool: 输入数据是否有效
        Raises:
            ValueError: 当输入数据无效时抛出异常
        '''
        if not (isinstance(rotations, np.ndarray) and isinstance(translations, np.ndarray)):
            raise ValueError("输入数据必须为numpy数组")
        if rotations.shape[0] != translations.shape[0]:
            raise ValueError("rotations和translations的帧数必须一致")

        # 检查第二个维度是否为3
        if rotations.shape[1] != 3 or translations.shape[1] != 3:
            raise ValueError("rotations和translations的第二个维度必须为3")

    # 速度、角度等的计算
    # * 1. 算旋转的角速度 & 角加速度(optional)
    def calculate_angular_speed(
        self, 
        rotations: np.ndarray, 
        input_representation: Literal["euler", "quaternion", "axis_angle"] = "axis_angle",
        calc_acceleration: bool = False,
        smooth: bool = False,
        smooth_window_size: int = 5,
        **kwargs,
    ) -> np.ndarray:
        '''
        计算旋转的角速度 & 角加速度
        Args:
            - rotations: 
                - (T, 3)的旋转数据, if input_representation=="euler"，则为欧拉角表示的旋转数据
                - (T, 4)的旋转数据, if input_representation=="quaternion"，则为四元数表示的旋转数据
                - (T, 3)的旋转数据, if input_representation=="axis_angle"，则为轴角表示的旋转数据
            - input_representation: 输入表示方法
            - calc_acceleration: 是否计算加速度
            - smooth: 是否对速度和加速度进行时间窗口平滑（Savitzky-Golay滤波）
            - smooth_window_size: 平滑窗口大小，必须为奇数，默认5
        Returns:
            - velocity: np.ndarray, (T-1, 3) or (T-1, 4)
                - 角速度：如果输入为欧拉角，则为每个轴的角速度；
                - 如果输入为四元数，则为四元数的变化率?
            - acceleration: np.ndarray, (T-2, 3) or (T-2, 4)，
                - 只有在calc_acceleration=True时才计算和返回，否则只返回角速度

        # TODO: 角加速度的计算方法需要进一步确定，特别是四元数表示的情况下，可能需要使用四元数微分的相关知识
        # TODO: 还需要考虑旋转表示的奇异性问题，例如欧拉角的万向锁问题，可能需要在计算前进行适当的转换或处理
        # TODO: 算速度/加速度之后， 维度会减少， 是否补0来保持与原数据的一致？
        '''
        
        raise NotImplementedError("calculate_angular_speed method is under development, wait to be implemented")

        # 统一转化为 四元数 表示来计算角速度
        if input_representation == "euler":
            order = kwargs.get('euler_order', 'xyz')
            rotations_quat = R.from_euler(order, rotations, degrees=True).as_quat()  # (T, 4)
        elif input_representation == "axis_angle":
            # rotations_quat = PoseMath.axis_angle_to_quat(rotations)  # (T, 4)
            rotations_quat = R.from_rotvec(rotations, degrees=True).as_quat()  # (T, 4)
        
        quat = PoseMath.ensure_quat_continuity(rotations_quat)
        # 计算角速度矢量

        q_prev = quat[:-1]
        q_curr = quat[1:]
        q_rel = PoseMath.quat_mul(q_curr, PoseMath.quat_conj(q_prev))
        q_rel = PoseMath.normalize_quaternion(q_rel)

        # 将相邻帧间的四元数差转换为轴角矢量 (T-1, 3)
        # 轴角矢量将旋转轴和旋转角编码在一个3D向量中
        axis_angle_diff = R.from_quat(q_rel).as_rotvec()  # (T-1, 3)
        
        # 平滑轴角矢量（如果启用）——对矢量的三个分量分别平滑
        if smooth and len(axis_angle_diff) >= smooth_window_size:
            smooth_ws = smooth_window_size
            if smooth_ws % 2 == 0:
                smooth_ws += 1  # 保证窗口大小为奇数
            smooth_ws = min(smooth_ws, len(axis_angle_diff))
            if smooth_ws >= 3:
                polyorder = min(2, smooth_ws - 1)
                # 对三个分量分别应用平滑滤波
                axis_angle_diff = np.column_stack([
                    savgol_filter(axis_angle_diff[:, i], smooth_ws, polyorder)
                    for i in range(3)
                ])
        
        # 从平滑后的轴角矢量计算速度（矢量模长乘以fps）
        angle_magnitude = np.linalg.norm(axis_angle_diff, axis=1)  # (T-1,)
        vel = np.zeros(quat.shape[0], dtype=np.float64)
        vel[1:] = angle_magnitude * self.sequence_fps
        vel[0] = vel[1]

        speed = {
            'velocity': vel
        }

        if calc_acceleration:
            acc = np.zeros(quat.shape[0], dtype=np.float64)
            acc[1:] = np.abs(np.diff(vel) * self.sequence_fps)
            acc[0] = acc[1]
            
            # 平滑加速度（如果启用）
            if smooth and len(acc) >= smooth_window_size:
                smooth_ws = smooth_window_size
                if smooth_ws % 2 == 0:
                    smooth_ws += 1
                smooth_ws = min(smooth_ws, len(acc))
                if smooth_ws >= 3:
                    polyorder = min(2, smooth_ws - 1)
                    acc = savgol_filter(acc, smooth_ws, polyorder)
            
            speed['acceleration'] = acc
        
        return speed

    # * 2. 算平移的线速度 & 加速度(optional)
    def calculate_linear_speed(
        self, 
        translations: np.ndarray,
        calc_acceleration: bool = False,
        smooth: bool = False,
        smooth_window_size: int = 5,
    ) -> np.ndarray:
        '''
        计算平移的线速度 & 加速度
        Args:
            - translations: (T, 3)的平移数据
            - calc_acceleration: 是否计算加速度
            - smooth: 是否对速度和加速度进行时间窗口平滑（Savitzky-Golay滤波）
            - smooth_window_size: 平滑窗口大小，必须为奇数，默认5
        Returns:
            - velocity: np.ndarray, (T-1, 3)的线速度数据
            - acceleration: np.ndarray, (T-2, 3)的线加速度数据，只有在calc_acceleration=True时才计算和返回
        '''
        t = translations.shape[0]
        if t == 1:
            return np.zeros(1, dtype=np.float64)

        diff = np.diff(translations.astype(np.float64), axis=0)
        reduce_axes = tuple(range(1, diff.ndim))
        dist = np.linalg.norm(diff, axis=reduce_axes)

        vel = np.zeros(t, dtype=np.float64)
        vel[1:] = dist * self.sequence_fps
        vel[0] = vel[1]

        # 平滑速度（如果启用）
        if smooth and len(vel) >= smooth_window_size:
            if smooth_window_size % 2 == 0:
                smooth_window_size += 1  # 保证窗口大小为奇数
            smooth_window_size = min(smooth_window_size, len(vel))
            if smooth_window_size >= 3:
                polyorder = min(2, smooth_window_size - 1)
                vel = savgol_filter(vel, smooth_window_size, polyorder)

        speed = {
            'velocity': vel,
        }

        if calc_acceleration:
            acc = np.zeros(t, dtype=np.float64)
            acc[1:] = np.abs(np.diff(vel) * self.sequence_fps)
            acc[0] = acc[1]
            
            # 平滑加速度（如果启用）
            if smooth and len(acc) >= smooth_window_size:
                if smooth_window_size % 2 == 0:
                    smooth_window_size += 1
                smooth_window_size = min(smooth_window_size, len(acc))
                if smooth_window_size >= 3:
                    polyorder = min(2, smooth_window_size - 1)
                    acc = savgol_filter(acc, smooth_window_size, polyorder)
            
            speed['acceleration'] = acc

        return speed

    # 离群值检测
    def detect_pose_outliers(
        self, 
        rotations: np.ndarray, 
        translations: np.ndarray, 
        mode: dict,
    ) -> dict:
        '''
        检测姿态数据中的异常值
        Args:
            rotations: (T, 3)的旋转数据
            translations: (T, 3)的平移数据
            mode: 离群检测方法和参数，详见父类Filter.detect_outliers的mode参数说明
        Returns:
            dict: 包含旋转和平移离群检测结果的字典，格式如下：
                {
                    'rotation_outliers': {
                        'threshold': ...,
                        'std': ...,
                        'MAD': ...,
                        'ml': ...,
                    },
                    'translation_outliers': {
                        'threshold': ...,
                        'std': ...,
                        'MAD': ...,
                        'ml': ...,
                    }
                }
        '''
        # TODO 传入参数有点怪

    # 插值修复

    # 可视化对比

    # 统计分析与导出

def batch_analysis(
    path_pattern: str, 
    seqID_index: int = 0, # 0 表示文件名即为 sequence ID, 1 表示父目录名为 sequence ID， 以此类推
    save_root: str = 'output/debug/vis',
):
    input_paths = glob.glob(path_pattern)
    input_paths = sorted(input_paths)  # 确保输入文件顺序一致
    for input_path in tqdm(input_paths):
        fname_ext = os.path.basename(input_path)
        dot_index = fname_ext.rfind('.')
        fname, ext = fname_ext[:dot_index], fname_ext[dot_index:]
        
        tmp_path = input_path
        for _ in range(seqID_index):
            tmp_path = os.path.dirname(tmp_path)
        seq_id = os.path.basename(tmp_path)
        single_analysis(
            input_path=input_path,
            seq_ID=seq_id,
            output_root=save_root,
        )

        # break # for debug, process only one file

def single_analysis(
    input_path: str,
    seq_ID: str, 
    output_root: str,
):
    smplh_data = np.load(
        input_path, 
        allow_pickle=True,
    )
    global_orient = smplh_data['root_orient']  # (N, 3)
    global_trans  = smplh_data['trans']       # (N, 3)
    fps = 30.0
    
    print(f'global_orient.shape={global_orient.shape} global_trans.shape={global_trans.shape} fps={fps}')

    save_dir = f'{output_root}/{seq_ID}'
    os.makedirs(save_dir, exist_ok=True)

    pf = PoseFilter(sequence_fps=fps)
    pf.check_input(global_orient, global_trans)
    speeds = {}
    # speeds['angular'] = pf.calculate_angular_speed(
    #     global_orient, 
    #     input_representation="axis_angle", calc_acceleration=True, 
    #     smooth=True, 
    #     smooth_window_size=5,
    # )
    speeds['linear'] = pf.calculate_linear_speed(
        global_trans, 
        calc_acceleration=True, 
        smooth=True, 
        smooth_window_size=5, 
    )
    
    for speed_type in speeds.keys():
        speed = speeds[speed_type]
        for property_name in speed.keys():
            val = speed[property_name]
            mode = {
                    'MAD': {
                        'mad_multiplier': 3.0,
                    }, 
                    'ml': {
                        'model_type': 'isolation_forest',
                        'model_params': {
                            'contamination': 'auto',
                        }
                    }
                }
            outliers_dict = pf.detect_outliers(
                val, 
                mode = mode, 
            )

            print("Outliers detected:")
            for key, value in outliers_dict.items():
                print(f"  {key}: {np.sum(value['mask'])}")

                # 可视化
                visualize2d.visualization_2d(
                    data=val, 
                    path=f'{save_dir}/outliers-{speed_type}_{property_name}-{key}.png', 
                    title=f'{speed_type.capitalize()} Velocity with Outliers', 
                    mark_mask=outliers_dict['MAD']['mask'], 
                    # default_bounds=[-180, 180], 
                    # y_lines=[-90, 0, 90], 
                    # y_colors=['r', 'g', 'r'], 
                )

if __name__ == "__main__":
    target_dir = 'output_submit/repair/repaired_zitai_batch1'
    save_dir = 'output/debug/vis_repair'
    batch_analysis(
        path_pattern=f'{target_dir}/*/*.npz', 
        seqID_index=1, 
        save_root=save_dir,
    )

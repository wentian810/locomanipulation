import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from utils.vis3d_utils import Scatter3DVisualizer

class visualize2d:
    def __init__(self):
        pass

    @staticmethod
    def visualization_2d(
        # self, 
        data: np.ndarray, 
        path: str, 
        title: str, 
        mark_mask: Optional[np.ndarray]=None,
        filter: callable=None, 
        default_bounds: list=None, 
        y_lines: list=None, 
        y_colors: list=None, 
    ):
        '''
        可视化数据和离群帧
        Args:
            data: 输入数据，形状为 (T, )，其中 T 是帧数
        '''
        lens = data.shape[0]
        data_cp = data.copy()
        if filter is not None:
            data_cp = filter(data_cp)

        if default_bounds is not None:
            upbound, lowbound = default_bounds
            data_cp = np.clip(data_cp, lowbound, upbound)
        
        plt.figure(figsize=(12, 6))
        plt.plot(
            range(lens), 
            data_cp, 
            marker='o', 
            color='b', 
            label='Data',
            alpha=0.7, 
        )
        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Value')

        if mark_mask is not None:
            # mask 为 True 的帧标记出 scatter
            print(f"Marking {np.sum(mark_mask)} outliers in the plot")
            plt.scatter(
                np.arange(lens)[mark_mask], 
                data_cp[mark_mask], 
                color='r', 
                s=50, 
                marker='x', 
                label='marked', 
                alpha=0.5, 
            )

        if y_lines is not None:
            y_colors = y_colors if y_colors is not None else ['r'] * len(y_lines)
            for y_line, y_color in zip(y_lines, y_colors):
                plt.axhline(y=y_line, color=y_color, linestyle='--')
        
        plt.grid()
        plt.savefig(path)
        plt.close()

    @staticmethod
    def visualization_3d(
        # self, 
        data: np.ndarray,
        mark_mask: Optional[np.ndarray]=None,
    ):
        '''
        将 (N, 3) 的 np.ndarray 数据可视化为3D曲线，并标记离群帧
            - data: (N, 3) 的输入数据
            - mark_mask: (N, ) 的布尔数组，True表示对应帧是离群帧
            - path: 保存3D可视化图像的路径
            - title: 图像标题

        '''
        visualizer = Scatter3DVisualizer(data, mask=mark_mask)
        visualizer.show()


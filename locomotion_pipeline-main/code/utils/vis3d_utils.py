import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, CheckButtons, Slider
import warnings
warnings.filterwarnings('ignore')

class Visualization3d:
    def __init__(self):
        pass

    @staticmethod
    def set_color(vertices, vertices_colors):
        if vertices_colors is not None:
            # if single color
            if isinstance(vertices_colors, (list, tuple, np.ndarray)) and len(vertices_colors) == 3:
                vertex_colors = np.tile(vertices_colors, (len(vertices), 1))
            # if per-vertex color
            elif isinstance(vertices_colors, np.ndarray) and vertices_colors.shape == vertices.shape:
                vertex_colors = vertices_colors
            # if just mask, shape=(N,)
            elif isinstance(vertices_colors, np.ndarray) and vertices_colors.shape == (len(vertices),):
                # mask to color, normalize to [0, 255]
                normalized_color = (vertices_colors - vertices_colors.min()) / (vertices_colors.max() - vertices_colors.min()) * 255
                vertex_colors = normalized_color.astype(np.uint8)
                # expand to RGB
                vertex_colors = np.stack([vertex_colors] * 3, axis=-1)
            else:
                raise ValueError("point_color must be a single RGB color or an array of shape (N, 3)")
        else:
            vertex_colors = None
        return vertex_colors

    @staticmethod
    def export_single_mesh(
        vertices, 
        faces, 
        export_path, 
        point_color=None,
        export_texture=False, 
        texture_image=None, 
        texture_coords=None, 
    ):
        """
        Export a single mesh to a file.

        Args:
            vertices (np.ndarray): Shape (N, 3) array of vertex positions.
            faces (np.ndarray): Shape (F, 3) array of face indices.
            export_path (str): Path to save the exported file.
            export_format (str): Format to export ('obj' or 'ply').
            export_texture (bool): Whether to include texture information.
            texture_image (np.ndarray): Texture image if export_texture is True.
            texture_coords (np.ndarray): Texture coordinates if export_texture is True.
        """
        vertex_colors = Visualization3d.set_color(vertices, point_color)

        # Create a Trimesh object
        mesh = trimesh.Trimesh(
            vertices=vertices, 
            faces=faces, 
            vertex_colors=vertex_colors
        )

        # Handle texture if needed
        if export_texture and texture_image is not None and texture_coords is not None:
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=texture_coords,
                image=texture_image
            )

        # Export the mesh
        mesh.export(export_path)

    @staticmethod
    def export_single_pc(
        vertices, 
        export_path, 
        point_color=None,
    ):
        """
        Export a single point cloud to a file.

        Args:
            vertices (np.ndarray): Shape (N, 3) array of vertex positions.
            export_path (str): Path to save the exported file.
            point_color (np.ndarray): Optional shape (N, 3) array of RGB colors for each vertex.
            export_format (str): Format to export ('ply' or 'xyz').
        """
        # Create a Trimesh PointCloud object
        vertex_colors = Visualization3d.set_color(vertices, point_color)

        # Export the point cloud
        point_cloud = trimesh.points.PointCloud(
            vertices=vertices, 
            colors=vertex_colors, 
        )

        point_cloud.export(export_path)

    def export_sequence(
        self, 

    ):
        '''
        not implemented yet.
        '''
        pass

class Scatter3DVisualizer:
    def __init__(self, data, mask=None):
        """
        初始化3D散点图可视化器
        
        Parameters:
        -----------
        data : numpy.ndarray, shape (N, 3)
            三维点云数据
        mask : numpy.ndarray, shape (N,), optional
            布尔掩码，True 的点会被标记；不提供时默认全 False
        """
        self.data = data
        self.n_points = len(data)

        if mask is None:
            self.mask = np.zeros(self.n_points, dtype=bool)
        else:
            mask = np.asarray(mask)
            if mask.shape[0] != self.n_points:
                raise ValueError(f"mask length {mask.shape[0]} does not match data length {self.n_points}")
            self.mask = mask.astype(bool)
        
        # 创建图形和轴
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置图形标题和标签
        self.ax.set_title('3D Scatter Plot with Mask Visualization', fontsize=16)
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        
        # 颜色定义
        self.colors = {
            'mask_true': 'red',
            'mask_false': 'blue',
            'highlight': 'yellow'
        }
        
        # 点的大小
        self.point_size = 20
        self.highlight_size = 50
        
        # 透明度
        self.alpha_true = 0.8
        self.alpha_false = 0.4
        
        # 初始化绘图
        self.scatter_true = None
        self.scatter_false = None
        self.highlight_points = None
        
        # 创建交互控件
        self.create_widgets()
        
        # 绘制初始图形
        self.plot_scatter()
        
        plt.tight_layout()
        
    def plot_scatter(self):
        """绘制散点图"""
        # 清除之前的绘图
        self.ax.clear()
        
        # 设置坐标轴标签
        self.ax.set_xlabel('X', fontsize=12)
        self.ax.set_ylabel('Y', fontsize=12)
        self.ax.set_zlabel('Z', fontsize=12)
        
        # 分离mask为True和False的点
        true_points = self.data[self.mask]
        false_points = self.data[~self.mask]
        
        # 绘制mask为False的点（蓝色，半透明）
        if len(false_points) > 0:
            self.scatter_false = self.ax.scatter(false_points[:, 0], 
                                                 false_points[:, 1], 
                                                 false_points[:, 2],
                                                 c=self.colors['mask_false'], 
                                                 s=self.point_size,
                                                 alpha=self.alpha_false,
                                                 label='Mask=False')
        
        # 绘制mask为True的点（红色，不透明）
        if len(true_points) > 0:
            self.scatter_true = self.ax.scatter(true_points[:, 0], 
                                                true_points[:, 1], 
                                                true_points[:, 2],
                                                c=self.colors['mask_true'], 
                                                s=self.point_size,
                                                alpha=self.alpha_true,
                                                label='Mask=True')
        
        # 添加图例
        self.ax.legend(loc='upper right', fontsize=10)
        
        # 设置视图角度
        self.ax.view_init(elev=20, azim=45)
        
        # 重新绘制
        self.fig.canvas.draw_idle()
    
    def highlight_by_index(self, indices):
        """高亮显示指定索引的点"""
        # 移除之前的高亮
        if self.highlight_points is not None:
            self.highlight_points.remove()
        
        # 高亮新的点
        if len(indices) > 0:
            highlight_data = self.data[indices]
            self.highlight_points = self.ax.scatter(highlight_data[:, 0],
                                                    highlight_data[:, 1],
                                                    highlight_data[:, 2],
                                                    c=self.colors['highlight'],
                                                    s=self.highlight_size,
                                                    alpha=1.0,
                                                    edgecolors='black',
                                                    linewidth=2,
                                                    label='Highlighted')
            self.ax.legend(loc='upper right', fontsize=10)
        
        self.fig.canvas.draw_idle()
    
    def create_widgets(self):
        """创建交互控件"""
        # 设置控件位置
        plt.subplots_adjust(left=0.05, bottom=0.15, right=0.95, top=0.95)
        
        # 创建按钮轴
        button_ax = plt.axes([0.8, 0.05, 0.15, 0.04])
        self.button = Button(button_ax, 'Reset View', color='lightgray', hovercolor='0.975')
        self.button.on_clicked(self.reset_view)
        
        # 创建复选框轴
        checkbox_ax = plt.axes([0.02, 0.3, 0.15, 0.15])
        checkbox_labels = ['Show True Mask', 'Show False Mask', 'Show Grid']
        checkbox_status = [True, True, True]
        self.checkbox = CheckButtons(checkbox_ax, checkbox_labels, checkbox_status)
        self.checkbox.on_clicked(self.toggle_visibility)
        
        # 创建透明度滑块
        alpha_ax = plt.axes([0.02, 0.15, 0.15, 0.03])
        self.alpha_slider = Slider(alpha_ax, 'False Alpha', 0.0, 1.0, 
                                   valinit=self.alpha_false, valstep=0.05)
        self.alpha_slider.on_changed(self.update_alpha)
        
        # 创建点大小滑块
        size_ax = plt.axes([0.02, 0.08, 0.15, 0.03])
        self.size_slider = Slider(size_ax, 'Point Size', 5, 100, 
                                  valinit=self.point_size, valstep=5)
        self.size_slider.on_changed(self.update_size)
        
        # 添加鼠标点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
    
    def reset_view(self, event):
        """重置视图角度"""
        self.ax.view_init(elev=20, azim=45)
        self.fig.canvas.draw_idle()
    
    def toggle_visibility(self, label):
        """切换可见性"""
        if label == 'Show True Mask':
            if self.scatter_true is not None:
                self.scatter_true.set_visible(not self.scatter_true.get_visible())
        elif label == 'Show False Mask':
            if self.scatter_false is not None:
                self.scatter_false.set_visible(not self.scatter_false.get_visible())
        elif label == 'Show Grid':
            self.ax.grid(not self.ax._gridOn)
        
        self.fig.canvas.draw_idle()
    
    def update_alpha(self, val):
        """更新透明度"""
        self.alpha_false = val
        if self.scatter_false is not None:
            self.scatter_false.set_alpha(val)
        self.fig.canvas.draw_idle()
    
    def update_size(self, val):
        """更新点的大小"""
        self.point_size = val
        if self.scatter_true is not None:
            self.scatter_true.set_sizes([val])
        if self.scatter_false is not None:
            self.scatter_false.set_sizes([val])
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        """鼠标点击事件：高亮最近的点"""
        if event.inaxes == self.ax:
            # 获取点击位置的3D坐标
            click_point = np.array([event.xdata, event.ydata, 0])
            
            # 计算所有点到点击位置的距离（在XY平面）
            distances = np.sqrt(np.sum((self.data[:, :2] - click_point[:2])**2, axis=1))
            
            # 找到最近的点
            nearest_idx = np.argmin(distances)
            min_distance = distances[nearest_idx]
            
            # 如果距离小于阈值，高亮该点
            if min_distance < 0.5:
                self.highlight_by_index([nearest_idx])
                print(f"Highlighted point {nearest_idx}: {self.data[nearest_idx]}, Mask={self.mask[nearest_idx]}")
            else:
                self.highlight_by_index([])
    
    def show(self):
        """显示图形"""
        plt.show()

def generate_sample_data(n_points=200):
    """生成示例数据"""
    np.random.seed(42)
    
    # 生成三维点云数据
    data = np.random.randn(n_points, 3)
    
    # 生成mask：标记在某个区域内的点
    # 例如：标记x>0且y>0且z>0的点
    mask = (data[:, 0] > 0) & (data[:, 1] > 0) & (data[:, 2] > 0)
    
    # 添加一些额外的True点，使分布更均衡
    mask[50:80] = True
    mask[150:170] = False
    
    return data, mask

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    N = 300
    data, mask = generate_sample_data(N)
    
    print(f"数据形状: {data.shape}")
    print(f"Mask形状: {mask.shape}")
    print(f"Mask=True的点数: {np.sum(mask)}")
    print(f"Mask=False的点数: {np.sum(~mask)}")
    
    # 创建可视化器
    visualizer = Scatter3DVisualizer(data, mask)
    
    # 显示图形
    visualizer.show()
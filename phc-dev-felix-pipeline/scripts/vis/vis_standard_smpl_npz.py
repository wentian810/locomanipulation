import sys
import time
import numpy as np
import open3d as o3d
import torch
from smpl_sim.smpllib.smpl_parser import SMPL_Parser

npz_path = sys.argv[1]
data = np.load(npz_path, allow_pickle=True)

poses = data["poses"].astype(np.float32)          # (N, 24, 3)
trans = data["trans"].astype(np.float32)          # (N, 3)
betas = data["betas"].astype(np.float32).reshape(-1)
gender = str(data["gender"])

if gender == "male":
    smpl_parser = SMPL_Parser(model_path="data/smpl", gender="male")
elif gender == "female":
    smpl_parser = SMPL_Parser(model_path="data/smpl", gender="female")
else:
    smpl_parser = SMPL_Parser(model_path="data/smpl", gender="neutral")

with torch.no_grad():
    verts, joints = smpl_parser.get_joints_verts(
        pose=torch.from_numpy(poses.reshape(poses.shape[0], 72)),
        th_trans=torch.from_numpy(trans),
        th_betas=torch.from_numpy(betas[None, :]),
    )

verts = verts.numpy()
faces = smpl_parser.faces

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts[0])
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
vis.add_geometry(mesh)

paused = False
frame = 0
fps = 30.0

def toggle_pause(_):
    global paused
    paused = not paused
    return True

vis.register_key_callback(32, toggle_pause)  # space

while True:
    step_start = time.time()
    if not paused:
        mesh.vertices = o3d.utility.Vector3dVector(verts[frame % len(verts)])
        mesh.compute_vertex_normals()
        vis.update_geometry(mesh)
        frame += 1
    vis.poll_events()
    vis.update_renderer()
    elapsed = time.time() - step_start
    remaining = (1.0 / fps) - elapsed
    if remaining > 0:
        time.sleep(remaining)

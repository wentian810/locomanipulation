#!/usr/bin/env python3
"""Verify the packaged chair primitives collide in Isaac Gym PhysX.

This deliberately uses no PHC policy or humanoid asset: a dynamic sphere is
dropped onto the exact seat centre and onto a nearby floor-only control point.
If the seat probe stops well above the control probe, the scene collision mesh
is physically solid and correctly placed in the Z-up simulation frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaacgym import gymapi


def box_triangles(center: np.ndarray, extents: np.ndarray, rotation: np.ndarray):
    corners = np.asarray(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    vertices = center + (corners * extents) @ rotation.T
    triangles = np.asarray(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.uint32,
    )
    return np.ascontiguousarray(vertices, dtype=np.float32), triangles


def position_from_state(state) -> np.ndarray:
    pose = state["pose"][0]
    return np.asarray([pose["p"]["x"], pose["p"]["y"], pose["p"]["z"]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sphere-radius", type=float, default=0.05)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    payload = json.loads(args.primitives.read_text(encoding="utf-8"))
    if payload.get("frame") != "mujoco_world_z_up":
        raise ValueError(f"Unexpected primitive frame: {payload.get('frame')!r}")
    primitives = payload.get("primitives", [])
    seat = next(item for item in primitives if item.get("name") == "seat_support")
    center = np.asarray(seat["center"], dtype=np.float32)
    extents = np.asarray(seat["extents"], dtype=np.float32)
    rotation = np.asarray(seat["rotation_matrix"], dtype=np.float32)

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 120.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Isaac Gym could not create a PhysX simulation")

    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane.static_friction = 1.0
    plane.dynamic_friction = 1.0
    gym.add_ground(sim, plane)

    for primitive in primitives:
        if primitive.get("type") != "box":
            raise ValueError(f"Unsupported primitive type: {primitive.get('type')!r}")
        vertices, triangles = box_triangles(
            np.asarray(primitive["center"], dtype=np.float32),
            np.asarray(primitive["extents"], dtype=np.float32),
            np.asarray(primitive["rotation_matrix"], dtype=np.float32),
        )
        mesh_params = gymapi.TriangleMeshParams()
        mesh_params.nb_vertices = len(vertices)
        mesh_params.nb_triangles = len(triangles)
        mesh_params.transform.p = gymapi.Vec3(0.0, 0.0, 0.0)
        mesh_params.static_friction = 1.0
        mesh_params.dynamic_friction = 1.0
        mesh_params.restitution = 0.0
        gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), mesh_params)

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)
    asset_options = gymapi.AssetOptions()
    asset_options.density = 1000.0
    sphere = gym.create_sphere(sim, float(args.sphere_radius), asset_options)

    # The control begins outside the seat footprint but at the same horizontal
    # world height.  It must settle on the ground plane rather than the chair.
    seat_start = center + np.asarray([0.0, 0.0, 0.70], dtype=np.float32)
    control_start = center + rotation[:, 0] * (extents[0] * 1.25)
    control_start[2] = seat_start[2]
    actors = {}
    for name, point in (("seat", seat_start), ("floor_control", control_start)):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*map(float, point))
        actor = gym.create_actor(env, sphere, pose, name, 0, 0)
        shape = gym.get_actor_rigid_shape_properties(env, actor)
        shape[0].friction = 1.0
        shape[0].restitution = 0.0
        gym.set_actor_rigid_shape_properties(env, actor, shape)
        gym.set_rigid_body_color(env, actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
        actors[name] = actor

    gym.prepare_sim(sim)
    sample_steps = {
        int(round(fraction * args.seconds / sim_params.dt)): f"t={fraction * args.seconds:.2f}s"
        for fraction in (0.10, 0.25, 0.50, 1.00)
    }
    samples = {}
    for step in range(1, int(round(args.seconds / sim_params.dt)) + 1):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        if step in sample_steps:
            samples[sample_steps[step]] = {
                name: position_from_state(
                    gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
                ).round(6).tolist()
                for name, actor in actors.items()
            }

    final = {
        name: position_from_state(gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS))
        for name, actor in actors.items()
    }
    expected_seat_top_z = float(center[2] + 0.5 * extents[2])
    clearance = float(final["seat"][2] - final["floor_control"][2])
    passed = bool(clearance >= 0.20 and final["seat"][2] >= expected_seat_top_z)
    report = {
        "schema_version": 1,
        "primitive_frame": payload["frame"],
        "seat_center_z_up_m": center.astype(float).tolist(),
        "seat_extents_m": extents.astype(float).tolist(),
        "sphere_radius_m": float(args.sphere_radius),
        "expected_seat_top_world_z_m": expected_seat_top_z,
        "final_sphere_center_z_up_m": {key: value.round(6).tolist() for key, value in final.items()},
        "trajectory_samples_z_up_m": samples,
        "seat_minus_floor_center_height_m": round(clearance, 6),
        "pass": passed,
        "criterion": "seat probe remains >=20cm above floor control and above the seat top",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    gym.destroy_sim(sim)
    if not passed:
        raise SystemExit("FAIL: semantic chair seat did not support the dynamic probe")


if __name__ == "__main__":
    main()

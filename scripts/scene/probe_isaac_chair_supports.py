#!/usr/bin/env python3
"""Verify V19 chair seat and backrest collision in Isaac Gym PhysX.

The packaged PHC collision JSON must contain the stable-basis seat and the
world-identical stable-basis backrest. A vertical drop checks the seat; a
horizontal launch checks that the backrest cannot be traversed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaacgym import gymapi


_TRIANGLES = np.asarray(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.uint32,
)


def box_triangles(item: dict) -> tuple[np.ndarray, np.ndarray]:
    extents = np.asarray(item["extents"], dtype=np.float32)
    center = np.asarray(item["center"], dtype=np.float32)
    rotation = np.asarray(item["rotation_matrix"], dtype=np.float32)
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
    return np.ascontiguousarray(vertices), _TRIANGLES


def body_position(gym, env, actor) -> np.ndarray:
    state = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)
    point = state["pose"][0]["p"]
    return np.asarray([point["x"], point["y"], point["z"]], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primitives", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sphere-radius", type=float, default=0.05)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--backrest-speed", type=float, default=2.0)
    args = parser.parse_args()

    payload = json.loads(args.primitives.read_text(encoding="utf-8"))
    if payload.get("frame") != "mujoco_world_z_up":
        raise ValueError("chair primitives must use mujoco_world_z_up")
    indexed = {item.get("name"): item for item in payload.get("primitives", [])}
    missing = {"seat_support", "backrest_support"} - set(indexed)
    if missing:
        raise ValueError(f"V19 requires support primitives: {sorted(missing)}")
    seat = indexed["seat_support"]
    backrest = indexed["backrest_support"]
    seat_center = np.asarray(seat["center"], dtype=np.float64)
    seat_extents = np.asarray(seat["extents"], dtype=np.float64)
    seat_rotation = np.asarray(seat["rotation_matrix"], dtype=np.float64)
    back_center = np.asarray(backrest["center"], dtype=np.float64)
    back_extents = np.asarray(backrest["extents"], dtype=np.float64)
    back_rotation = np.asarray(backrest["rotation_matrix"], dtype=np.float64)
    back_normal = back_rotation[:, 0]

    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = 1.0 / 120.0
    params.substeps = 2
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    params.physx.solver_type = 1
    params.physx.num_position_iterations = 8
    params.physx.num_velocity_iterations = 1
    params.physx.num_threads = 4
    params.physx.use_gpu = True
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError("Isaac Gym could not create a PhysX simulation")
    try:
        ground = gymapi.PlaneParams()
        ground.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        ground.static_friction = 1.0
        ground.dynamic_friction = 1.0
        gym.add_ground(sim, ground)
        for primitive in payload["primitives"]:
            vertices, triangles = box_triangles(primitive)
            mesh = gymapi.TriangleMeshParams()
            mesh.nb_vertices = len(vertices)
            mesh.nb_triangles = len(triangles)
            mesh.static_friction = 1.0
            mesh.dynamic_friction = 1.0
            gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), mesh)

        env = gym.create_env(
            sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1
        )
        sphere = gym.create_sphere(sim, float(args.sphere_radius), gymapi.AssetOptions())
        seat_start = seat_center + np.asarray([0.0, 0.0, 0.70])
        floor_start = seat_center + seat_rotation[:, 0] * (seat_extents[0] * 1.25)
        floor_start[2] = seat_start[2]
        back_start = back_center - back_normal * (
            0.5 * back_extents[0] + args.sphere_radius + 0.12
        )
        actors = {}
        for name, point in (
            ("seat", seat_start),
            ("floor_control", floor_start),
            ("backrest", back_start),
        ):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*map(float, point))
            actors[name] = gym.create_actor(env, sphere, pose, name, 0, 0)

        gym.prepare_sim(sim)
        state = gym.get_actor_rigid_body_states(env, actors["backrest"], gymapi.STATE_ALL)
        state["vel"]["linear"][0] = tuple(float(args.backrest_speed) * back_normal)
        gym.set_actor_rigid_body_states(env, actors["backrest"], state, gymapi.STATE_ALL)

        max_back_signed = float(np.dot(back_start - back_center, back_normal))
        for _ in range(int(round(args.seconds / params.dt))):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            point = body_position(gym, env, actors["backrest"])
            max_back_signed = max(
                max_back_signed, float(np.dot(point - back_center, back_normal))
            )
        final = {name: body_position(gym, env, actor) for name, actor in actors.items()}
        seat_clearance = float(final["seat"][2] - final["floor_control"][2])
        expected_seat_top = float(seat_center[2] + 0.5 * seat_extents[2])
        seat_pass = bool(
            seat_clearance >= 0.20 and final["seat"][2] >= expected_seat_top
        )
        backrest_pass = bool(max_back_signed <= 0.5 * args.sphere_radius)
        report = {
            "schema_version": 1,
            "primitive_frame": payload["frame"],
            "tests": {
                "seat_vertical_drop": {
                    "pass": seat_pass,
                    "seat_minus_floor_center_height_m": round(seat_clearance, 6),
                    "expected_seat_top_world_z_m": expected_seat_top,
                },
                "backrest_horizontal_launch": {
                    "pass": backrest_pass,
                    "normal_world": back_normal.round(6).tolist(),
                    "start_signed_distance_m": round(
                        float(np.dot(back_start - back_center, back_normal)), 6
                    ),
                    "max_signed_distance_m": round(max_back_signed, 6),
                    "crossing_threshold_m": round(0.5 * args.sphere_radius, 6),
                },
            },
            "final_sphere_centers_z_up_m": {
                name: point.round(6).tolist() for name, point in final.items()
            },
            "pass": bool(seat_pass and backrest_pass),
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["pass"]:
            raise SystemExit("FAIL: V19 chair support probe did not pass")
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()


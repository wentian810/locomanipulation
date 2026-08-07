import argparse
import copy
import os
import pathlib
import pickle
import tempfile
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = pathlib.Path(__file__).parent
ASSET_ROOT = HERE / ".." / "assets"
PIPELINE_ROOT = HERE.parents[1]
DEFAULT_SHARPA_ROOT = (
    PIPELINE_ROOT
    / "do-as-i-do-main"
    / "retargeting"
    / "retargeting"
    / "assets"
    / "robots"
    / "sharpa"
)
DEFAULT_BRAINCO_ROOT = (
    HERE
    / ".."
    / "third_party"
    / "robot_hands"
    / "brainco_description"
    / "revo2_system"
)

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "unitree_g1_with_hands": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof_with_hands.xml",
    "unitree_h1": ASSET_ROOT / "unitree_h1" / "h1.xml",
    "unitree_h1_with_hand": ASSET_ROOT / "unitree_h1" / "h1_with_hand.xml",
    "unitree_h1_with_hand_wrist": ASSET_ROOT / "unitree_h1" / "h1_with_hand.xml",
    "unitree_h1_2": ASSET_ROOT / "unitree_h1_2" / "h1_2_handless.xml",
    "booster_t1": ASSET_ROOT / "booster_t1" / "T1_serial.xml",
    "booster_t1_29dof": ASSET_ROOT / "booster_t1_29dof" / "t1_mocap.xml",
    "stanford_toddy": ASSET_ROOT / "stanford_toddy" / "toddy_mocap.xml",
    "fourier_n1": ASSET_ROOT / "fourier_n1" / "n1_mocap.xml",
    "engineai_pm01": ASSET_ROOT / "engineai_pm01" / "pm_v2.xml",
    "kuavo_s45": ASSET_ROOT / "kuavo_s45" / "biped_s45_collision.xml",
    "hightorque_hi": ASSET_ROOT / "hightorque_hi" / "hi_25dof.xml",
    "galaxea_r1pro": ASSET_ROOT / "galaxea_r1pro" / "r1_pro.xml",
    "berkeley_humanoid_lite": ASSET_ROOT / "berkeley_humanoid_lite" / "bhl_scene.xml",
    "booster_k1": ASSET_ROOT / "booster_k1" / "K1_serial.xml",
    "pnd_adam_lite": ASSET_ROOT / "pnd_adam_lite" / "scene.xml",
    "tienkung": ASSET_ROOT / "tienkung" / "mjcf" / "tienkung.xml",
    "pal_talos": ASSET_ROOT / "pal_talos" / "talos.xml",
    "fourier_gr3": ASSET_ROOT / "fourier_gr3v2_1_1" / "mjcf" / "gr3v2_1_1_dummy_hand.xml",
}

ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_g1_with_hands": "pelvis",
    "unitree_h1": "pelvis",
    "unitree_h1_with_hand": "pelvis",
    "unitree_h1_with_hand_wrist": "pelvis",
    "unitree_h1_2": "pelvis",
    "booster_t1": "Waist",
    "booster_t1_29dof": "Waist",
    "stanford_toddy": "waist_link",
    "fourier_n1": "base_link",
    "engineai_pm01": "LINK_BASE",
    "kuavo_s45": "base_link",
    "hightorque_hi": "base_link",
    "galaxea_r1pro": "torso_link4",
    "berkeley_humanoid_lite": "imu_2",
    "booster_k1": "Trunk",
    "pnd_adam_lite": "pelvis",
    "tienkung": "Base_link",
    "pal_talos": "base_link",
    "fourier_gr3": "base_link",
}


CAMERA_PRESETS = {
    "threequarter": (12.0, 135.0),
    "front": (5.0, 180.0),
    "side": (5.0, 90.0),
    "back": (5.0, 0.0),
    "top": (75.0, 90.0),
}


def open_video_writer(video_path, fps):
    return imageio.get_writer(
        str(video_path),
        format="FFMPEG",
        fps=fps,
        macro_block_size=1,
    )


def load_motion(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    fps = float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0])
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    local_body_pos = data.get("local_body_pos")
    if local_body_pos is not None:
        local_body_pos = np.asarray(local_body_pos, dtype=np.float32)
    body_names = data.get("link_body_list")
    return data, fps, root_pos, root_rot_xyzw, dof_pos, local_body_pos, body_names


def quat_xyzw_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def parse_xml_body_tree(xml_path):
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in {xml_path}")
    xml_body_root = worldbody.find("body")
    if xml_body_root is None:
        raise ValueError(f"root body not found in {xml_path}")

    body_names = []
    parent_indices = []

    def add_body(node, parent_index):
        body_index = len(body_names)
        body_names.append(node.attrib.get("name", f"body_{body_index}"))
        parent_indices.append(parent_index)
        for child in node.findall("body"):
            add_body(child, body_index)

    add_body(xml_body_root, -1)
    return body_names, np.asarray(parent_indices, dtype=np.int32)


def map_parent_indices(parsed_names, parsed_parents, motion_body_names):
    if not motion_body_names:
        return parsed_parents
    name_to_motion_idx = {name: i for i, name in enumerate(motion_body_names)}
    parents = np.full(len(motion_body_names), -1, dtype=np.int32)
    for parsed_idx, name in enumerate(parsed_names):
        motion_idx = name_to_motion_idx.get(name)
        if motion_idx is None:
            continue
        parsed_parent = parsed_parents[parsed_idx]
        if parsed_parent >= 0:
            parent_name = parsed_names[parsed_parent]
            parents[motion_idx] = name_to_motion_idx.get(parent_name, -1)
    return parents


def world_body_positions(root_pos, root_rot_xyzw, local_body_pos):
    rot = quat_xyzw_to_matrix(root_rot_xyzw)
    return root_pos[:, None, :] + np.einsum("tij,tbj->tbi", rot, local_body_pos)


def z_yaw_matrix(degrees):
    radians = np.deg2rad(degrees)
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)
    return np.asarray(
        [
            [cos_v, -sin_v, 0.0],
            [sin_v, cos_v, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_camera_path(args):
    if not args.camera_path:
        return None

    camera_file = pathlib.Path(args.camera_path)
    camera_data = np.load(camera_file)
    if args.camera_path_source == "isaac":
        pos = np.asarray(camera_data["camera_pos_isaac"], dtype=np.float32)
        target = np.asarray(camera_data["camera_target_isaac"], dtype=np.float32)
        subject = np.asarray(
            camera_data.get("subject_isaac", target),
            dtype=np.float32,
        )
    else:
        pos = np.asarray(camera_data["camera_pos_world"], dtype=np.float32)
        target = np.asarray(camera_data["camera_target_world"], dtype=np.float32)
        subject = np.asarray(
            camera_data.get("subject_world", target),
            dtype=np.float32,
        )
        world_to_isaac = np.asarray(
            camera_data.get("world_to_isaac", [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            dtype=np.float32,
        )
        pos = pos @ world_to_isaac.T
        target = target @ world_to_isaac.T
        subject = subject @ world_to_isaac.T

    if args.camera_yaw_offset_deg:
        yaw = z_yaw_matrix(args.camera_yaw_offset_deg)
        pos = pos @ yaw.T
        target = target @ yaw.T
        subject = subject @ yaw.T

    horizontal_fov_deg = None
    if "horizontal_fov_deg" in camera_data:
        horizontal_fov_deg = float(np.asarray(camera_data["horizontal_fov_deg"]).reshape(-1)[0])

    return {
        "pos": pos,
        "target": target,
        "subject": subject,
        "horizontal_fov_deg": horizontal_fov_deg,
        "path": str(camera_file),
    }


def camera_sample_index(frame_idx, motion_frames, camera_frames):
    if camera_frames <= 1 or motion_frames <= 1:
        return 0
    return int(round(frame_idx * (camera_frames - 1) / (motion_frames - 1)))


def camera_params_from_pos_target(pos, target, fallback_distance, fallback_elevation, fallback_azimuth):
    delta = np.asarray(pos, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return fallback_distance, fallback_elevation, fallback_azimuth

    horizontal = float(np.linalg.norm(delta[:2]))
    elevation = float(np.rad2deg(np.arctan2(-delta[2], max(horizontal, 1e-8))))
    azimuth = float(np.rad2deg(np.arctan2(-delta[1], -delta[0])))
    return distance, elevation, azimuth


def vertical_fovy_from_horizontal(horizontal_fov_deg, width, height):
    horizontal = np.deg2rad(horizontal_fov_deg)
    vertical = 2.0 * np.arctan(np.tan(horizontal / 2.0) * (height / width))
    return float(np.rad2deg(vertical))


def parse_rgba(value):
    if not value:
        return None
    parts = [p for p in str(value).replace(",", " ").split() if p]
    if len(parts) != 4:
        raise ValueError(f"--robot_rgba expects 4 values, got {value!r}")
    rgba = np.asarray([float(p) for p in parts], dtype=np.float32)
    return np.clip(rgba, 0.0, 1.0)


def parse_rgb(value):
    if not value:
        return None
    parts = [p for p in str(value).replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(f"--background_rgb expects 3 values, got {value!r}")
    rgb = np.asarray([float(p) for p in parts], dtype=np.float32)
    if np.max(rgb) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def apply_robot_rgba(model, rgba):
    if rgba is None:
        return
    visible = model.geom_rgba[:, 3] > 0.0
    model.geom_rgba[visible] = rgba


def apply_object_rgba(model, rgba):
    if rgba is None:
        return
    import mujoco as mj

    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "tracked_object_proxy_geom")
    if geom_id >= 0:
        model.geom_rgba[geom_id] = rgba


def replace_dark_background(frame, rgb, threshold):
    if rgb is None:
        return frame
    frame = np.asarray(frame).copy()
    mask = np.max(frame, axis=-1) <= int(threshold)
    frame[mask] = rgb
    return frame


def parse_vec(value, count, default):
    if not value:
        return list(default)
    parts = [p for p in str(value).replace(",", " ").split() if p]
    if len(parts) != count:
        raise ValueError(f"Expected {count} values, got {value!r}")
    return [float(p) for p in parts]


def find_body(root, name):
    for body in root.iter("body"):
        if body.attrib.get("name") == name:
            return body
    return None


def remove_render_only_keyframes(root):
    """Remove base-robot keyframes after adding external hand joints.

    The temporary merged MJCF is used only for per-frame kinematic rendering,
    where qpos is assigned from the body and hand motion arrays.  A base XML
    keyframe still contains the pre-merge qpos width, so MuJoCo rejects it as
    soon as the two Sharpa chains add their joints.
    """
    for keyframe in list(root.findall("keyframe")):
        root.remove(keyframe)


def ensure_sharpa_assets(target_root, sharpa_root, scale=1.0):
    asset = target_root.find("asset")
    if asset is None:
        asset = ET.SubElement(target_root, "asset")
    existing = {item.attrib.get("name") for item in asset.findall("mesh")}
    for side in ("left", "right"):
        side_root = ET.parse(sharpa_root / f"{side}.xml").getroot()
        for mesh in side_root.find("asset").findall("mesh"):
            name = mesh.attrib.get("name")
            if not name or name in existing:
                continue
            item = copy.deepcopy(mesh)
            item.set("file", str((sharpa_root / "meshes" / mesh.attrib["file"]).resolve()))
            item.set("scale", f"{scale} {scale} {scale}")
            asset.append(item)
            existing.add(name)


def _scale_xml_numeric_attribute(element, attribute, factor):
    """Scale one whitespace-separated numeric MJCF attribute in place."""

    value = element.get(attribute)
    if value is None:
        return
    try:
        values = [float(token) for token in value.replace(",", " ").split()]
    except ValueError as exc:
        raise ValueError(
            f"Cannot scale {element.tag}.{attribute}={value!r} in Sharpa MJCF"
        ) from exc
    element.set(attribute, " ".join(f"{number * factor:.8g}" for number in values))


def scale_sharpa_kinematic_subtree(body, scale):
    """Uniformly scale an imported Sharpa hand around its mount frame.

    Scaling the STL assets alone shrinks each link mesh while leaving the
    body-to-body offsets unchanged.  The result is a small palm connected by
    visually long, thin fingers.  Keep every local length in the imported
    kinematic subtree in the same unit system as the meshes instead.
    """

    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Sharpa scale must be finite and positive, got {scale!r}")
    if np.isclose(scale, 1.0):
        return

    for element in body.iter():
        # ``pos`` is a local Cartesian offset for bodies, joints, sites,
        # inertials, cameras and geoms.  The root body is still at its source
        # origin here; its host-robot mount position is assigned afterwards.
        _scale_xml_numeric_attribute(element, "pos", scale)
        if element.tag == "geom":
            _scale_xml_numeric_attribute(element, "fromto", scale)
            _scale_xml_numeric_attribute(element, "size", scale)
        elif element.tag == "site":
            _scale_xml_numeric_attribute(element, "size", scale)
        elif element.tag == "inertial":
            # Rendering is kinematic, but preserving mass/inertia dimensions
            # keeps the temporary MJCF physically self-consistent as well.
            _scale_xml_numeric_attribute(element, "mass", scale ** 3)
            _scale_xml_numeric_attribute(element, "diaginertia", scale ** 5)
            _scale_xml_numeric_attribute(element, "fullinertia", scale ** 5)


def ensure_brainco_assets_and_defaults(target_root, brainco_root):
    asset = target_root.find("asset")
    if asset is None:
        asset = ET.SubElement(target_root, "asset")
    existing = {item.attrib.get("name") for item in asset.findall("mesh")}

    target_default = target_root.find("default")
    if target_default is None:
        target_default = ET.SubElement(target_root, "default")
    existing_classes = {
        item.attrib.get("class")
        for item in target_default.findall("default")
        if item.attrib.get("class")
    }

    for side in ("left", "right"):
        source_xml = brainco_root / "mjcf" / f"revo2_{side}.xml"
        side_root = ET.parse(source_xml).getroot()
        source_asset = side_root.find("asset")
        if source_asset is not None:
            for mesh in source_asset.findall("mesh"):
                name = mesh.attrib.get("name")
                if not name or name in existing:
                    continue
                item = copy.deepcopy(mesh)
                item.set(
                    "file",
                    str((source_xml.parent / mesh.attrib["file"]).resolve()),
                )
                asset.append(item)
                existing.add(name)
        source_default = side_root.find("default")
        if source_default is not None:
            for default_class in source_default.findall("default"):
                class_name = default_class.attrib.get("class")
                if not class_name or class_name in existing_classes:
                    continue
                target_default.append(copy.deepcopy(default_class))
                existing_classes.add(class_name)


def sanitize_external_sharpa_body(body):
    """Prevent host-robot defaults from changing imported Sharpa joints.

    Some Unitree XMLs set actuator-force limits in the root ``<default>``.  The
    copied Sharpa kinematic sidecar has many joints but no per-joint actuator
    force ranges, so inheriting that default makes MuJoCo reject the merged
    model.  Keep the host defaults for the host robot and opt the imported hand
    subtree out explicitly.
    """

    for joint in body.iter("joint"):
        joint.set("actuatorfrclimited", "false")


def strip_native_g1_hand_visuals(root):
    """Remove the stock G1 palm/finger visuals before mounting Sharpa.

    Keep the native bodies, joints and actuators intact so the source GMR qpos
    layout does not change.  Only geoms whose mesh/name starts with the G1 hand
    prefix are removed.  This is more reliable than changing alpha after model
    compilation because the stock G1 hand is rooted at ``wrist_yaw_link``, not
    at the legacy ``hand_link`` expected by the old hiding helper.
    """

    removed = 0
    for side in ("left", "right"):
        wrist = find_body(root, f"{side}_wrist_yaw_link")
        if wrist is None:
            continue
        prefix = f"{side}_hand_"
        for parent in wrist.iter():
            for child in list(parent):
                if child.tag != "geom":
                    continue
                mesh_name = child.attrib.get("mesh", "")
                geom_name = child.attrib.get("name", "")
                if mesh_name.startswith(prefix) or geom_name.startswith(prefix):
                    parent.remove(child)
                    removed += 1
    if removed:
        print(
            f"[render] Removed {removed} native G1 hand visual geoms; "
            "the Sharpa hand is the only visible hand model.",
            flush=True,
        )
    return removed


def build_unitree_sharpa_visual_xml(
    base_xml,
    sharpa_root,
    mount_pos,
    left_mount_quat,
    right_mount_quat,
    left_mount_pos=None,
    scale=1.0,
    right_mount_pos=None,
):
    base_xml = pathlib.Path(base_xml)
    sharpa_root = pathlib.Path(sharpa_root)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    strip_native_g1_hand_visuals(root)
    ensure_sharpa_assets(root, sharpa_root, scale)

    for side in ("left", "right"):
        side_tree = ET.parse(sharpa_root / f"{side}.xml")
        side_root = side_tree.getroot()
        source_body = side_root.find("worldbody").find("body")
        body = copy.deepcopy(source_body)
        scale_sharpa_kinematic_subtree(body, scale)
        sanitize_external_sharpa_body(body)
        side_mount_pos = (
            left_mount_pos
            if side == "left" and left_mount_pos is not None
            else right_mount_pos
            if side == "right" and right_mount_pos is not None
            else mount_pos
        )
        body.set("pos", " ".join(f"{v:.8g}" for v in side_mount_pos))
        mount_quat = left_mount_quat if side == "left" else right_mount_quat
        body.set("quat", " ".join(f"{v:.8g}" for v in mount_quat))
        # Legacy H1 already exposes a hand-link body at its flange. H1-2's
        # handless asset instead terminates at the 3-DoF wrist yaw link.
        parent = find_body(root, f"{side}_hand_link")
        if parent is None:
            parent = find_body(root, f"{side}_wrist_yaw_link")
        if parent is None:
            raise ValueError(
                f"Neither {side}_hand_link nor {side}_wrist_yaw_link "
                f"was found in {base_xml}"
            )
        parent.append(body)

    remove_render_only_keyframes(root)

    tmp = tempfile.NamedTemporaryFile(
        prefix="h1_sharpa_visual_",
        suffix=".xml",
        dir=str(base_xml.parent),
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name, encoding="unicode")
    return pathlib.Path(tmp.name)


# Backward-compatible import name used by the contact-analysis utilities.
build_h1_sharpa_visual_xml = build_unitree_sharpa_visual_xml


def build_g1_brainco_visual_xml(
    base_xml,
    brainco_root,
    left_mount_pos,
    right_mount_pos,
    left_mount_quat,
    right_mount_quat,
):
    """Mount the official BrainCo Revo2 MJCF pair on a handless Unitree G1."""
    base_xml = pathlib.Path(base_xml)
    brainco_root = pathlib.Path(brainco_root)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    ensure_brainco_assets_and_defaults(root, brainco_root)

    for side in ("left", "right"):
        source_xml = brainco_root / "mjcf" / f"revo2_{side}.xml"
        side_root = ET.parse(source_xml).getroot()
        source_body = side_root.find("worldbody").find("body")
        body = copy.deepcopy(source_body)
        mount_pos = left_mount_pos if side == "left" else right_mount_pos
        mount_quat = left_mount_quat if side == "left" else right_mount_quat
        body.set("pos", " ".join(f"{value:.8g}" for value in mount_pos))
        body.set("quat", " ".join(f"{value:.8g}" for value in mount_quat))
        parent = find_body(root, f"{side}_wrist_yaw_link")
        if parent is None:
            raise ValueError(f"{side}_wrist_yaw_link not found in {base_xml}")
        parent.append(body)

    remove_render_only_keyframes(root)

    tmp = tempfile.NamedTemporaryFile(
        prefix="g1_brainco_revo2_visual_",
        suffix=".xml",
        dir=str(base_xml.parent),
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name, encoding="unicode")
    return pathlib.Path(tmp.name)


def qpos_joint_names(model):
    ordered = []
    for joint_id in range(model.njnt):
        name = mj_name(model, "joint", joint_id)
        if name:
            ordered.append((int(model.jnt_qposadr[joint_id]), name))
    return [name for _, name in sorted(ordered)]


def mj_name(model, obj_type, obj_id):
    import mujoco as mj

    obj = {
        "joint": mj.mjtObj.mjOBJ_JOINT,
        "body": mj.mjtObj.mjOBJ_BODY,
        "geom": mj.mjtObj.mjOBJ_GEOM,
    }[obj_type]
    return mj.mj_id2name(model, obj, obj_id)


def joint_qpos_map(model):
    mapping = {}
    for joint_id in range(model.njnt):
        name = mj_name(model, "joint", joint_id)
        if name:
            mapping[name] = int(model.jnt_qposadr[joint_id])
    return mapping


def assign_named_qpos(data, qpos_map, names, values):
    for name, value in zip(names, np.asarray(values).reshape(-1)):
        adr = qpos_map.get(str(name))
        if adr is not None:
            data.qpos[adr] = float(value)


def hide_original_h1_hands(model):
    import mujoco as mj

    hand_ids = {}
    for side in ("left", "right"):
        body_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_BODY,
            f"{side}_hand_link",
        )
        if body_id >= 0:
            hand_ids[side] = body_id
    # H1-2 uses a handless base asset, so there is no native hand to hide.
    if not hand_ids:
        return
    sharpa_ids = {
        "left": model.body("left_base_tx").id,
        "right": model.body("right_base_tx").id,
    }

    def has_ancestor(body_id, ancestor_id):
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        for side in hand_ids:
            if has_ancestor(body_id, hand_ids[side]) and not has_ancestor(body_id, sharpa_ids[side]):
                model.geom_rgba[geom_id, 3] = 0.0
                break


def hide_original_g1_rubber_hands(model):
    for side in ("left", "right"):
        try:
            body_id = model.body(f"{side}_rubber_hand").id
        except KeyError:
            continue
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == body_id:
                model.geom_rgba[geom_id, 3] = 0.0


def load_external_hand_motion(path):
    if not path:
        return None
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {
        "path": path,
        "left_names": [str(x) for x in data["left_hand_qpos_names"]],
        "right_names": [str(x) for x in data["right_hand_qpos_names"]],
        "left_qpos": np.asarray(data["left_hand_qpos"], dtype=np.float32),
        "right_qpos": np.asarray(data["right_hand_qpos"], dtype=np.float32),
    }


def scale_sharpa_root_translation_qpos(external_motion, scale):
    """Scale Sharpa's three free palm translations with its geometry.

    The six root qpos values in a Sharpa track place the standalone hand model
    in the source palm frame.  Once the attached hand geometry is scaled, the
    translation part must be scaled too; articulation angles are unitless and
    intentionally remain unchanged.
    """

    if external_motion is None:
        return None
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Sharpa scale must be finite and positive, got {scale!r}")
    if np.isclose(scale, 1.0):
        return external_motion

    for side in ("left", "right"):
        names = external_motion[f"{side}_names"]
        qpos = np.asarray(external_motion[f"{side}_qpos"], dtype=np.float32).copy()
        indices = [
            index
            for index, name in enumerate(names)
            if name in {f"{side}_pos_x", f"{side}_pos_y", f"{side}_pos_z"}
        ]
        if indices:
            qpos[:, indices] *= scale
        external_motion[f"{side}_qpos"] = qpos
    return external_motion


def parse_geom_size(value, object_type):
    if value is None:
        if object_type == "sphere":
            return np.asarray([0.12], dtype=np.float32)
        if object_type in {"cylinder", "capsule"}:
            return np.asarray([0.08, 0.18], dtype=np.float32)
        return np.asarray([0.14, 0.09, 0.07], dtype=np.float32)
    size = np.asarray(value, dtype=np.float32).reshape(-1)
    if object_type == "sphere":
        return size[:1]
    if object_type in {"cylinder", "capsule"}:
        if size.size == 1:
            return np.asarray([size[0], size[0] * 2.0], dtype=np.float32)
        return size[:2]
    if size.size == 1:
        return np.repeat(size[:1], 3).astype(np.float32)
    if size.size == 2:
        return np.asarray([size[0], size[1], size[1]], dtype=np.float32)
    return size[:3]


def npz_value(data, key, default=None):
    return data[key] if key in data else default


def load_object_motion(path):
    if not path:
        return None
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}

    pos_key = "position" if "position" in data else "positions"
    if pos_key not in data:
        raise ValueError(f"{path} must contain position or positions")
    position = np.asarray(data[pos_key], dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"object position must have shape (T, 3), got {position.shape}")

    if "quat_wxyz" in data:
        quat = np.asarray(data["quat_wxyz"], dtype=np.float32)
    elif "quat_xyzw" in data:
        quat_xyzw = np.asarray(data["quat_xyzw"], dtype=np.float32)
        quat = quat_xyzw[:, [3, 0, 1, 2]]
    else:
        quat = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (position.shape[0], 1))
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"object quat must have shape (T, 4), got {quat.shape}")
    quat = quat / np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)
    valid = np.asarray(
        npz_value(data, "valid", np.ones(position.shape[0], dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if valid.shape[0] != position.shape[0]:
        raise ValueError(
            f"object valid must have {position.shape[0]} entries, got {valid.shape}"
        )

    object_type = str(np.asarray(npz_value(data, "object_type", "box")).reshape(-1)[0])
    geom_size = parse_geom_size(npz_value(data, "geom_size", npz_value(data, "size")), object_type)
    rgba = np.asarray(npz_value(data, "rgba", [0.95, 0.62, 0.16, 0.9]), dtype=np.float32).reshape(-1)
    if rgba.size != 4:
        rgba = np.asarray([0.95, 0.62, 0.16, 0.9], dtype=np.float32)
    visual_mesh_path = ""
    if "visual_mesh_path" in data:
        visual_mesh_path = str(np.asarray(data["visual_mesh_path"]).reshape(-1)[0])
        visual_mesh_path = pathlib.Path(visual_mesh_path)
        if not visual_mesh_path.is_absolute():
            visual_mesh_path = (path.parent / visual_mesh_path).resolve()
        if not visual_mesh_path.exists():
            raise FileNotFoundError(visual_mesh_path)
    mesh_scale = np.asarray(
        npz_value(data, "mesh_scale", [1.0, 1.0, 1.0]),
        dtype=np.float32,
    ).reshape(-1)
    if mesh_scale.size == 1:
        mesh_scale = np.repeat(mesh_scale, 3)
    if mesh_scale.size != 3 or np.any(mesh_scale <= 0):
        raise ValueError(f"mesh_scale must contain three positive values, got {mesh_scale}")
    if object_type == "mesh" and not visual_mesh_path:
        raise ValueError(f"{path} declares object_type=mesh without visual_mesh_path")
    collision_mesh_paths = []
    for value in np.asarray(
        npz_value(data, "collision_mesh_paths", []),
        dtype=str,
    ).reshape(-1):
        collision_path = pathlib.Path(str(value))
        if not collision_path.is_absolute():
            collision_path = (path.parent / collision_path).resolve()
        if collision_path.exists():
            collision_mesh_paths.append(collision_path)
    density = float(
        np.asarray(npz_value(data, "density", 600.0)).reshape(-1)[0]
    )
    mass_kg = float(
        np.asarray(npz_value(data, "mass_kg", 0.0)).reshape(-1)[0]
    )
    center_of_mass = np.asarray(
        npz_value(data, "center_of_mass_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(-1)
    diaginertia = np.asarray(
        npz_value(data, "diaginertia_kg_m2", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(-1)
    inertial_quat = np.asarray(
        npz_value(data, "inertial_quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(-1)
    physics_asset_mode = str(
        np.asarray(
            npz_value(data, "physics_asset_mode", "visual_only")
        ).reshape(-1)[0]
    )
    if physics_asset_mode == "physics":
        if not collision_mesh_paths:
            raise ValueError("physics object has no collision mesh paths")
        if not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError("physics object mass must be positive")
        if (
            center_of_mass.shape != (3,)
            or diaginertia.shape != (3,)
            or inertial_quat.shape != (4,)
            or not np.isfinite(center_of_mass).all()
            or not np.isfinite(diaginertia).all()
            or not np.isfinite(inertial_quat).all()
            or np.any(diaginertia <= 0)
            or np.linalg.norm(inertial_quat) <= 1e-8
        ):
            raise ValueError("physics object COM/inertia is invalid")
        inertial_quat = inertial_quat / np.linalg.norm(inertial_quat)
    friction = np.asarray(
        npz_value(data, "friction", [1.0, 0.05, 0.005]),
        dtype=np.float32,
    ).reshape(-1)
    solref = np.asarray(
        npz_value(data, "solref", [0.01, 1.0]),
        dtype=np.float32,
    ).reshape(-1)
    return {
        "path": path,
        "position": position,
        "quat_wxyz": quat,
        "valid": valid,
        "object_type": object_type,
        "geom_size": geom_size,
        "rgba": np.clip(rgba, 0.0, 1.0),
        "visual_mesh_path": visual_mesh_path,
        "mesh_scale": mesh_scale,
        "collision_mesh_paths": collision_mesh_paths,
        "density": density,
        "mass_kg": mass_kg,
        "center_of_mass_m": center_of_mass,
        "diaginertia_kg_m2": diaginertia,
        "inertial_quat_wxyz": inertial_quat,
        "physics_asset_mode": physics_asset_mode,
        "friction": friction,
        "solref": solref,
    }


def build_object_visual_xml(base_xml, object_motion):
    base_xml = pathlib.Path(base_xml)
    tree = ET.parse(base_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in {base_xml}")

    geom_attributes = {
        "name": "tracked_object_proxy_geom",
        "rgba": " ".join(f"{v:.8g}" for v in object_motion["rgba"]),
        "contype": "0",
        "conaffinity": "0",
    }
    if object_motion["object_type"] == "mesh":
        asset = root.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        mesh_name = "tracked_object_visual_mesh"
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": mesh_name,
                "file": str(object_motion["visual_mesh_path"]),
                "scale": " ".join(
                    f"{value:.8g}" for value in object_motion["mesh_scale"]
                ),
            },
        )
        geom_attributes.update({"type": "mesh", "mesh": mesh_name})
    else:
        geom_attributes.update(
            {
                "type": object_motion["object_type"],
                "size": " ".join(
                    f"{v:.8g}" for v in object_motion["geom_size"]
                ),
            }
        )

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "tracked_object_proxy",
            "mocap": "true",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
        },
    )
    ET.SubElement(
        body,
        "geom",
        geom_attributes,
    )

    tmp = tempfile.NamedTemporaryFile(
        prefix="object_proxy_visual_",
        suffix=".xml",
        dir=str(base_xml.parent),
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name, encoding="unicode")
    return pathlib.Path(tmp.name)


def build_static_scene_visual_xml(base_xml, scene_mujoco_xml):
    """Append a packaged static-scene MJCF to a robot visual MJCF.

    The scene package owns all mesh geometry.  Its XML is intentionally a
    standalone worldbody, while GMR owns the robot worldbody, so this function
    imports only the scene assets and static bodies.  Mesh paths are made
    absolute before writing a temporary combined XML beside the robot asset;
    consequently the published package remains portable and is never edited.
    """
    base_xml = pathlib.Path(base_xml)
    scene_mujoco_xml = pathlib.Path(scene_mujoco_xml)
    if not scene_mujoco_xml.is_file():
        raise FileNotFoundError(f"Scene MJCF does not exist: {scene_mujoco_xml}")

    tree = ET.parse(base_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in {base_xml}")
    scene_tree = ET.parse(scene_mujoco_xml)
    scene_root = scene_tree.getroot()
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise ValueError(f"worldbody not found in scene MJCF {scene_mujoco_xml}")

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    scene_asset = scene_root.find("asset")
    if scene_asset is not None:
        for child in scene_asset:
            imported = copy.deepcopy(child)
            if imported.tag == "mesh" and imported.get("file"):
                mesh_path = pathlib.Path(imported.attrib["file"])
                if not mesh_path.is_absolute():
                    mesh_path = scene_mujoco_xml.parent / mesh_path
                if not mesh_path.is_file():
                    raise FileNotFoundError(
                        "Scene MJCF mesh is missing: "
                        f"{mesh_path} (referenced by {scene_mujoco_xml})"
                    )
                imported.set("file", mesh_path.resolve().as_posix())
            asset.append(imported)
    for child in scene_worldbody:
        worldbody.append(copy.deepcopy(child))

    tmp = tempfile.NamedTemporaryFile(
        prefix="static_scene_visual_",
        suffix=".xml",
        dir=str(base_xml.parent),
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name, encoding="unicode")
    return pathlib.Path(tmp.name)


def set_equal_axes(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(-0.05, center[2] - radius), center[2] + radius)
    ax.set_box_aspect([1, 1, 1])


def render_skeleton(args):
    _, fps, root_pos, root_rot, _, local_body_pos, motion_body_names = load_motion(args.robot_motion_path)
    if local_body_pos is None:
        raise ValueError("Motion file has no local_body_pos. Re-run smpl_npz_to_robot_headless.py to generate it.")

    xml_path = pathlib.Path(args.robot_xml) if args.robot_xml else ROBOT_XML_DICT[args.robot]
    parsed_names, parsed_parents = parse_xml_body_tree(xml_path)
    parents = map_parent_indices(parsed_names, parsed_parents, motion_body_names)
    body_pos = world_body_positions(root_pos, root_rot, local_body_pos)

    frame_indices = list(
        range(max(0, args.start_frame), body_pos.shape[0], max(1, args.skip))
    )
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    video_fps = args.fps if args.fps > 0 else max(1, int(round(fps / max(1, args.skip))))

    pathlib.Path(args.video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(args.video_path, video_fps)
    fig = plt.figure(figsize=(args.width / 100, args.height / 100), dpi=100)

    for out_idx, frame_idx in enumerate(frame_indices):
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        pts = body_pos[frame_idx]
        for child_idx, parent_idx in enumerate(parents):
            if parent_idx < 0 or child_idx >= pts.shape[0] or parent_idx >= pts.shape[0]:
                continue
            a, b = pts[parent_idx], pts[child_idx]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#2563eb", linewidth=2.0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, color="#dc2626", alpha=0.8)
        center_idx = 0
        if motion_body_names:
            try:
                center_idx = motion_body_names.index(ROBOT_BASE_DICT.get(args.robot, motion_body_names[0]))
            except ValueError:
                center_idx = 0
        set_equal_axes(ax, pts[center_idx], args.radius)
        ax.view_init(elev=args.elevation, azim=args.azimuth)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{args.robot} | frame {frame_idx}/{body_pos.shape[0]}")
        fig.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(image)
        if (out_idx + 1) % 30 == 0:
            print(f"Rendered {out_idx + 1}/{len(frame_indices)} frames", flush=True)

    writer.close()
    plt.close(fig)
    print(f"Saved skeleton video to {args.video_path}", flush=True)


def render_mujoco(args):
    if args.mujoco_gl:
        mujoco_gl = str(args.mujoco_gl).strip().lower()
        os.environ["MUJOCO_GL"] = mujoco_gl
        if mujoco_gl in {"egl", "osmesa"} and not os.environ.get("PYOPENGL_PLATFORM"):
            os.environ["PYOPENGL_PLATFORM"] = mujoco_gl

    import mujoco as mj

    motion_data, fps, root_pos, root_rot_xyzw, dof_pos, _, _ = load_motion(args.robot_motion_path)
    base_xml_path = pathlib.Path(args.robot_xml) if args.robot_xml else ROBOT_XML_DICT[args.robot]
    if args.sharpa_hand_npz and args.brainco_hand_npz:
        raise ValueError("Sharpa and BrainCo external hands cannot be enabled together")
    external_hand_model = ""
    external_hand_path = ""
    if args.sharpa_hand_npz:
        external_hand_model = "sharpa"
        external_hand_path = args.sharpa_hand_npz
    elif args.brainco_hand_npz:
        external_hand_model = "brainco_revo2"
        external_hand_path = args.brainco_hand_npz
    external_motion = load_external_hand_motion(external_hand_path)
    if external_hand_model == "sharpa":
        external_motion = scale_sharpa_root_translation_qpos(external_motion, 1.0)
    object_motion = load_object_motion(args.object_motion_path)
    temp_xml_paths = []
    source_dof_names = None
    if external_motion is not None:
        source_model = mj.MjModel.from_xml_path(str(base_xml_path))
        source_dof_names = motion_data.get('dof_names', qpos_joint_names(source_model))
        if external_hand_model == "sharpa":
            mount_pos = parse_vec(args.sharpa_mount_pos, 3, [0.055, 0.0, 0.0])
            left_mount_pos = parse_vec(args.sharpa_left_mount_pos, 3, mount_pos)
            right_mount_pos = parse_vec(args.sharpa_right_mount_pos, 3, mount_pos)
            legacy_mount_quat = parse_vec(args.sharpa_mount_quat, 4, [0.5, 0.5, 0.5, 0.5])
            # Sharpa's left/right assets are mirrored across their local Y axis.
            # H1's hand-link frames are not mirrored, so using the right-hand
            # mount for both hands reverses the left index/pinky and thumb side.
            left_mount_quat = parse_vec(
                args.sharpa_left_mount_quat,
                4,
                [0.5, -0.5, 0.5, -0.5] if not args.sharpa_mount_quat else legacy_mount_quat,
            )
            right_mount_quat = parse_vec(
                args.sharpa_right_mount_quat,
                4,
                legacy_mount_quat,
            )
            xml_path = build_unitree_sharpa_visual_xml(
                base_xml_path,
                args.sharpa_root,
                mount_pos,
                left_mount_quat,
                right_mount_quat,
                left_mount_pos=left_mount_pos,
                right_mount_pos=right_mount_pos,
                scale=1.0,
            )
        else:
            xml_path = build_g1_brainco_visual_xml(
                base_xml_path,
                args.brainco_root,
                parse_vec(args.brainco_left_mount_pos, 3, [0.0415, 0.003, 0.0]),
                parse_vec(args.brainco_right_mount_pos, 3, [0.0415, -0.003, 0.0]),
                parse_vec(args.brainco_left_mount_quat, 4, [0.0, 0.0, 0.70710678, -0.70710678]),
                parse_vec(args.brainco_right_mount_quat, 4, [0.0, 0.0, 0.70710678, -0.70710678]),
            )
        temp_xml_paths.append(xml_path)
    else:
        xml_path = base_xml_path
    if object_motion is not None:
        xml_path = build_object_visual_xml(xml_path, object_motion)
        temp_xml_paths.append(xml_path)
    if args.scene_mujoco_xml:
        xml_path = build_static_scene_visual_xml(xml_path, args.scene_mujoco_xml)
        temp_xml_paths.append(xml_path)
    model = mj.MjModel.from_xml_path(str(xml_path))
    apply_robot_rgba(model, parse_rgba(args.robot_rgba))
    if object_motion is not None:
        apply_object_rgba(model, object_motion["rgba"])
    if external_hand_model == "sharpa":
        hide_original_h1_hands(model)
        hide_original_g1_rubber_hands(model)
    elif external_hand_model == "brainco_revo2":
        hide_original_g1_rubber_hands(model)
    background_rgb = parse_rgb(args.background_rgb)
    camera_path = load_camera_path(args)
    if camera_path and args.camera_subject_align_xy:
        subject = camera_path["subject"]
        aligned_subject = np.stack(
            [
                subject[
                    camera_sample_index(
                        frame_idx,
                        root_pos.shape[0],
                        subject.shape[0],
                    )
                ]
                for frame_idx in range(root_pos.shape[0])
            ],
            axis=0,
        )
        root_pos = root_pos.copy()
        root_pos[:, :2] = aligned_subject[:, :2]
    if camera_path and args.camera_fov_from_path and camera_path["horizontal_fov_deg"] is not None:
        model.vis.global_.fovy = vertical_fovy_from_horizontal(
            camera_path["horizontal_fov_deg"], args.width, args.height
        )
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=args.height, width=args.width)

    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.distance = args.radius
    cam.elevation = args.elevation
    cam.azimuth = args.azimuth

    frame_indices = list(
        range(max(0, args.start_frame), root_pos.shape[0], max(1, args.skip))
    )
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    video_fps = args.fps if args.fps > 0 else max(1, int(round(fps / max(1, args.skip))))

    pathlib.Path(args.video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(args.video_path, video_fps)

    base_name = ROBOT_BASE_DICT.get(args.robot)
    base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_name) if base_name else -1
    qpos_map = joint_qpos_map(model) if external_motion is not None else None
    object_mocap_id = -1
    if object_motion is not None:
        object_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "tracked_object_proxy")
        if object_body_id < 0:
            raise RuntimeError("tracked_object_proxy body missing after XML insertion")
        object_mocap_id = int(model.body_mocapid[object_body_id])
        if object_mocap_id < 0:
            raise RuntimeError("tracked_object_proxy is not a MuJoCo mocap body")

    for out_idx, frame_idx in enumerate(frame_indices):
        data.qpos[:3] = root_pos[frame_idx]
        data.qpos[3:7] = root_rot_xyzw[frame_idx][[3, 0, 1, 2]]
        if external_motion is None:
            data.qpos[7:] = dof_pos[frame_idx]
        else:
            assign_named_qpos(data, qpos_map, source_dof_names, dof_pos[frame_idx])
            hand_idx = camera_sample_index(frame_idx, root_pos.shape[0], external_motion["left_qpos"].shape[0])
            assign_named_qpos(data, qpos_map, external_motion["left_names"], external_motion["left_qpos"][hand_idx])
            assign_named_qpos(data, qpos_map, external_motion["right_names"], external_motion["right_qpos"][hand_idx])
        if object_motion is not None:
            object_idx = camera_sample_index(frame_idx, root_pos.shape[0], object_motion["position"].shape[0])
            if object_motion["valid"][object_idx]:
                data.mocap_pos[object_mocap_id] = object_motion["position"][object_idx]
                data.mocap_quat[object_mocap_id] = object_motion["quat_wxyz"][object_idx]
            else:
                data.mocap_pos[object_mocap_id] = [0.0, 0.0, -1000.0]
                data.mocap_quat[object_mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(model, data)
        if camera_path:
            if args.camera_static:
                camera_idx = 0
            else:
                camera_idx = camera_sample_index(frame_idx, root_pos.shape[0], camera_path["pos"].shape[0])
            target = camera_path["target"][camera_idx]
            distance, elevation, azimuth = camera_params_from_pos_target(
                camera_path["pos"][camera_idx],
                target,
                args.radius,
                args.elevation,
                args.azimuth,
            )
            cam.lookat[:] = target
            cam.distance = distance
            cam.elevation = elevation
            cam.azimuth = azimuth
        elif base_id >= 0:
            cam.lookat[:] = data.xpos[base_id]
        renderer.update_scene(data, camera=cam)
        frame = replace_dark_background(renderer.render(), background_rgb, args.background_threshold)
        writer.append_data(frame)
        if (out_idx + 1) % 30 == 0:
            print(f"Rendered {out_idx + 1}/{len(frame_indices)} frames", flush=True)

    writer.close()
    renderer.close()
    for path in temp_xml_paths:
        try:
            pathlib.Path(path).unlink()
        except OSError:
            pass
    print(f"Saved MuJoCo video to {args.video_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Render GMR robot motion on a headless server.")
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--robot_motion_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--mode", choices=["skeleton", "mujoco", "mesh"], default="skeleton")
    parser.add_argument("--robot_xml", default=None)
    parser.add_argument("--mujoco_gl", default="", help="Optional MuJoCo GL backend, e.g. egl or osmesa.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--radius", type=float, default=2.2)
    parser.add_argument("--camera_mode", choices=["custom", "threequarter", "front", "side", "back", "top"], default="threequarter")
    parser.add_argument("--elevation", type=float, default=12.0)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--camera_path", default="", help="Optional GVHMR camera npz to drive MuJoCo rendering.")
    parser.add_argument("--camera_path_source", choices=["isaac", "world"], default="isaac")
    parser.add_argument("--camera_yaw_offset_deg", type=float, default=0.0)
    parser.add_argument("--camera_fov_from_path", action="store_true")
    parser.add_argument("--camera_static", action="store_true",
                        help="Always use the first camera frame (like gvhmr_static in ISSAC GYM).")
    parser.add_argument(
        "--camera_subject_align_xy",
        action="store_true",
        help=(
            "For comparison rendering only, align robot root XY to the "
            "GVHMR subject trajectory stored in --camera_path."
        ),
    )
    parser.add_argument(
        "--robot_rgba",
        default="",
        help="Optional RGBA override for visible robot geoms, e.g. '0.72,0.74,0.76,1'.",
    )
    parser.add_argument(
        "--background_rgb",
        default="",
        help="Optional RGB replacement for near-black MuJoCo background, e.g. '245,246,248'.",
    )
    parser.add_argument("--background_threshold", type=int, default=4)
    parser.add_argument("--sharpa_hand_npz", default="", help="Optional 001_sharpa_hands.npz for visual Sharpa hands.")
    parser.add_argument("--sharpa_root", default=str(DEFAULT_SHARPA_ROOT), help="Directory containing Sharpa left.xml/right.xml.")
    parser.add_argument("--sharpa_mount_pos", default="0.055,0,0")
    parser.add_argument("--sharpa_left_mount_pos", default="")
    parser.add_argument("--sharpa_right_mount_pos", default="")
    parser.add_argument(
        "--sharpa_mount_quat",
        default="",
        help="Legacy shared Sharpa mount quaternion; prefer the side-specific options.",
    )
    parser.add_argument("--sharpa_left_mount_quat", default="")
    parser.add_argument("--sharpa_right_mount_quat", default="")
    parser.add_argument("--brainco_hand_npz", default="", help="Optional hardware-coupled Revo2 trajectory.")
    parser.add_argument("--brainco_root", default=str(DEFAULT_BRAINCO_ROOT), help="BrainCo revo2_system directory.")
    parser.add_argument("--brainco_left_mount_pos", default="0.0415,0.003,0")
    parser.add_argument("--brainco_right_mount_pos", default="0.0415,-0.003,0")
    parser.add_argument("--brainco_left_mount_quat", default="0,0,0.70710678,-0.70710678")
    parser.add_argument("--brainco_right_mount_quat", default="0,0,0.70710678,-0.70710678")
    parser.add_argument("--object_motion_path", default="", help="Optional object_motion.npz with position and quat_wxyz.")
    parser.add_argument(
        "--scene_mujoco_xml",
        default="",
        help="Optional static scene MJCF exported by the scene reconstruction package.",
    )
    args = parser.parse_args()

    if args.scene_mujoco_xml and args.camera_subject_align_xy:
        raise ValueError(
            "--camera_subject_align_xy is incompatible with --scene_mujoco_xml; "
            "a scene render must use the exact audited robot trajectory"
        )
    if args.camera_mode != "custom":
        args.elevation, args.azimuth = CAMERA_PRESETS[args.camera_mode]

    if args.mode in {"mujoco", "mesh"}:
        render_mujoco(args)
    else:
        render_skeleton(args)


if __name__ == "__main__":
    main()

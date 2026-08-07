# V24 MPC、SONIC 与 HoloMotion 物理坐姿迭代说明

更新日期：2026-08-07；适用分支：`scene-added`；人形流程基线：
`de65768739b747706106218c9a161a788d927b41`；场景接入提交：
`cba2452dd15c232faa564b1b5b4232c5eb89a102`。

## 1. 先回答“为什么没有 MPC 脚本”

MPC/MJPC、SONIC 和 v24 物理坐姿的**自研适配脚本已经上传**到
`scene-added`。此前容易造成误解，是因为真正执行 v24 通过样例的底层控制器
属于第三方 HoloMotion；该仓库在服务器上包含约 1.7 GB 的依赖、资产和
checkpoint，不能把整棵工作目录直接提交到 Git。

因此发布方式分成两层：

| 内容 | GitHub 中的位置 | 是否完整上传 |
| --- | --- | --- |
| MJPC/MPC 任务构造、约束、轨迹审计 | `scripts/scene/build_g1_mjpc_task_bundle.py`、`audit_g1_mpc_trajectory.py`、`audit_mjpc_chair_contact_types.py`、`diagnose_mjpc_chair_contacts.py` 等 | 是 |
| SONIC 静态场景接入、无重置回放、trace 审计 | `scripts/scene/prepare_sonic_*.py`、`run_sonic_audited_sim.py`、`analyze_sonic_trace.py` | 是 |
| v24 HoloMotion 参考构造、静态椅子、实际 rollout 审计及 2×2 组合 | `scripts/scene/*holomotion*`、`scripts/audit_holomotion_v24_*.py`、`scripts/compose_time_synchronised_v24_2x2.py` | 是 |
| HoloMotion 第三方代码 | `patches/holomotion_v24_evaluator.patch` + `patches/UPSTREAM_VERSIONS.md` | 以固定上游 revision + 本项目补丁交付 |
| HoloMotion checkpoint、VideoMimic 数据、渲染视频、`scene_work` | 不上传 | 否，属于大文件/运行产物 |

HoloMotion 固定到 `71ab7e976de23aa9bb351030dd61b05426d3443c`。补丁只改
`holomotion/src/evaluation/eval_mujoco_sim2sim.py`：记录语义椅子的真实接触
数量、座面法向力和最小距离，并支持受限的 post-seat 内部关节 PD 策略。它不
添加骨盆外力、mocap weld、逐帧状态重置或移动椅子。

重要区别：**MPC 已经被实际运行和审计过，但当前 v24 通过样例不是 MPC
通过的，而是冻结 HoloMotion ONNX 控制器在同一物理契约下通过的。** 不能把
“仓库里有 MPC 脚本”写成“最终结果由 MPC 得到”。

## 2. 本轮要解决的真实问题

目标从来不是把 GVHMR 的 SMPL、GMR 的 G1 和一把语义椅子叠加播放，而是：

1. GMR 输出仍然是动作参考；
2. 椅子在 MuJoCo 中是固定、可见、可碰撞的实体；
3. 机器人只在初始时刻写一次 `qpos/qvel`；
4. 后续状态仅由受力矩限制的关节控制和连续 `mj_step` 积分产生；
5. 通过接触力、最小几何距离、根部连续性和跟踪误差共同判定，而不是只看视频。

所有物理候选均禁止以下捷径：

```text
逐帧 qpos/qvel 重写
xfrc_applied 骨盆或 base 外力
mocap weld
运行时移动椅子或按帧调整椅子高度
为了“坐上去”而训练新的策略/强化学习
```

这也是为什么旧的 `mj_forward` 渲染和逐帧 `mj_step` 检查不能称为动态通过：
它们都可能在每个视频帧重新写入机器人姿态，接触并没有机会反作用到后续状态。

## 3. 版本前的根因复核：先停止修错对象

### 3.1 根部“跳变”不是当前 GMR 原始动作的固有问题

对 Buying、Dramatic、Pianist 当前使用的 `robot_motion.pkl` 逐帧检查后，root
的相邻帧位移约为 2–3 cm，没有超过 5 cm 的门槛。历史上出现过单帧约 12.04 cm
的 `static-hard` 分支；它以及相关旧渲染产物已被拒绝。

因此，早期视频中的高度跳变主要来自历史动作、场景、渲染产物混用，或者渲染端
额外覆盖 root，而不是当前 GMR 原始 root 序列突然跳变。为此补上了：

- `scripts/scene/audit_robot_motion_root.py`：动作根部连续性和输入 SHA 审计；
- `scripts/scene/write_gmr_render_contract.py`：动作、场景、渲染合同绑定；
- `scripts/run_pipeline_from_config.py` 与 GMR 渲染脚本：有 scene XML 时拒绝
  render-only 的 root XY 覆盖。

这一步只消除了“假跳变”的来源，不等于已经证明动态仿真中的脚不滑或机器人不跌倒。

### 3.2 脚的 9.5 cm 旧误差是审计错误，不是碰撞脚坐标偏移

曾报告左脚低于地面约 9.5 cm，复查发现审计脚本把不属于踝部的后代链一起计入
最低点。视觉 G1 网格与官方 Unitree 碰撞几何逐帧比对后，左脚最低点在第 0、90、179
帧均接近，初帧约为 0 mm 与 +1.5 mm。

所以当前没有证据支持“GMR 视觉脚和官方 G1 碰撞脚系统性错位”。动态阶段仍需用
`scripts/audit_holomotion_v24_foot_floor.py` 同时审计双脚对唯一 floor geom 的最小距离，
不能因为静态 FK 对齐就宣布脚的问题彻底解决。

### 3.3 真正的几何根因：SMPL 椅子尺度与固定 G1 形态不兼容

VideoMimic 恢复椅子时以 SMPL 人体尺度为参照；GMR 的 G1 则具有固定连杆长度。测量
表明 G1 大腿和小腿相对 SMPL 的度量比例约为 `0.792`。把人类尺寸椅子直接放进 G1
世界，会同时造成椅面高度、椅腿位置、髋部高度和足部可达性不一致。

旧的“把椅子硬平移到机器人臀部下方”的做法有时视觉上接近接触，但改变了椅子在
原视频中的位置，因此被拒绝。后续只允许把**椅子与相机一起**做一个全局静态 Sim(3)
变换，并用 `audit_similarity_projection_contract.py` 验证原视频投影不变；不允许按帧
移动椅子或 root。

## 4. 迭代过程和每一步的结果

### 4.1 场景证据、语义椅子与坐标合同

VideoMimic/MegaSAM/MegaHunter/NKSR 的原始网格保留为场景证据，但不能直接承担人景
接触：人体边缘深度污染、跨帧人影残留、深度不连续和远景点会把椅子与背景粘成不稳定
碰撞网格。因此流程明确区分：

| 产物 | 用途 | 是否直接作为物理碰撞 |
| --- | --- | --- |
| raw reconstruction mesh | 可追溯场景证据 | 否 |
| visual-clean mesh | 可视化和拟合检查 | 否 |
| human-swept residual | 诊断人体残影清理 | 否 |
| semantic chair primitives | 有证据约束的椅面、椅背、椅腿 | 是，但须经过签名距离审计 |

V19/V21 的局部椅背/椅面修正曾尝试通过改机器人 root 贴近椅子。它们说明语义椅子可以
被构造，但会把机器人形态差异误当作场景误差，不能扩展为逐帧生产逻辑。V22 后将 PHC
和 GMR 分开：复杂交叉腿等 PHC 不稳定动作不再覆盖 GMR 的保真参考。

### 4.2 0.866 全局尺度候选：投影正确，但运动学不可行，拒绝

全关节 Sim(3) 的 `0.866` 候选在图像投影上正确，却仍使原始 GMR 坐姿大面积撞入座面。
尝试足锁 IK 抬高骨盆后，脚位置约束、保持参考姿态和避免椅面/椅腿碰撞无法同时满足；
IK 回落后仍有约 3–7 cm 穿入。这一候选没有送入正式 MPC 渲染，也不作为 v24 基线。

### 4.3 0.792 下肢度量候选：选为静态物理场景，但暴露椅腿–下肢矛盾

`0.792` 候选通过了以下前置条件：

- 静态尺度与坐标合同；
- 原视频投影不变；
- G1 任务 XML 编译；
- 官方 Unitree 碰撞几何及单一 floor 接触合同。

它在坐下早期能找到较小的座面修复窗口，但后段语义椅子的前腿会撞到 G1 腿或踝部。
这是后来所有控制实验都必须面对的真实约束：不能再次通过下调椅子、关闭椅腿碰撞或
重写 root 来掩盖。

### 4.4 MuJoCo MJPC/MPC：完成真实连续受力测试，但 Buying 被拒绝

MPC 线不是训练。`build_g1_mjpc_task_bundle.py` 把原始 GMR 29-DoF 姿态、root 和 marker
写为参考任务，允许规划器用有界关节力矩追踪；`audit_g1_mpc_trajectory.py` 只审计实际
`mj_step` 产生的状态。

Buying 的一次完整任务在 `smpl_g1_metric_scene_v3/task_ground_contact_metric_raw_v1.xml`
中跑完 222 帧（7.37 秒）。结果如下：

| 指标 | 实测 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| root 最大单帧高度变化 | 3.95 cm | ≤ 4 cm | 通过 |
| 坐下前最大椅子穿入 | 2.99 mm | ≤ 5 mm | 通过 |
| root RMSE | 12.53 cm | ≤ 10 cm | 失败 |
| 坐下后座面支持帧比例 | 20.9% | ≥ 60% | 失败 |
| 坐下后最大椅子穿入 | 12.7 mm | ≤ 10 mm | 失败 |

进一步的接触审计表明，最大穿入来自 `torso_link -> seat_support_geom`，约 12.89 mm；
椅腿只出现一次约 1.47 mm 的轻微接触。更关键的是，root 在第 120 帧就开始相对参考
偏离超过 10 cm，到第 179 帧已约前偏 44 cm、下偏 15 cm。接近力矩上限的比例只有
约 1.27%，所以不能解释为“电机力矩不够”；更符合 MPC 跟踪代价/预测策略过早放弃参考
而连续下坠。

结论是：该 MPC 运行有价值，因为它排除了离散 root 跳变和简单预坐下穿入；但它没有
形成稳定坐姿，不能渲染成最终 2×2，也不能称为物理通过。

### 4.5 SONIC：接入成功，但现成策略不适用于此坐姿，拒绝

SONIC 是现成的全身控制/仿真部署框架，并非在本项目中训练的模型。接入内容包括：

- `prepare_sonic_scene_compat.py`、`prepare_sonic_physics_config.py`：把固定椅子和
  Unitree DDS/配置接到 SONIC；
- `write_sonic_playback_input.py`：将 GMR 参考转换为回放输入；
- `run_sonic_audited_sim.py`：移除交互 demo 在跌倒时 `mj_resetData` 的隐藏重置，改为
  记录失败并终止；
- `analyze_sonic_trace.py`：检查实际 trace。

SONIC 的官方 Unitree PD 控制和 `mj_step` 均被实际启动，但在“视频恢复的坐下/跌坐
参考 + 新静态椅子”组合下，受控下蹲测试约 0.98 秒便失稳。原因不是 SONIC 没启动，
而是其冻结策略没有对该参考和椅子接触进行训练或优化。

本项目明确**不以训练 SONIC 为下一步**：那会扩展成新的强化学习课题，且无法回答当前
“给定 GMR 参考能否物理坐下”的工程问题。因此 SONIC 代码保留为可复现实验和失败证据，
但未用于 v24 通过结果。

### 4.6 转向冻结 HoloMotion：不训练，保持连续物理约束

v24 最终采用冻结 HoloMotion ONNX `model_14000` 的 G1 跟踪器。它不是重新训练，也不是
每帧播放 qpos。流程为：

1. `export_gmr_motion_to_holomotion_reference.py` 仅把 GMR 29-DoF motion 转成 HoloMotion
   所需的 `ref_*` 格式并重采样；完整 link pose 始终由 MuJoCo FK 重算。
2. `build_holomotion_static_chair_scene.py` 将六个已拟合椅子 primitive 扁平化为
   `worldbody` 的静态 geom，而不是附加成新 body。这样不会改变冻结网络的 `model.nbody`
   输入维度，椅子仍可见且可碰撞。
3. `build_holomotion_physical_seat_reference.py` 在真实 MuJoCo 几何中定位最后一个无碰撞
   approach frame，再平滑过渡到已单独验证的重力坐姿 keyframe。它只修复**参考**，不在
   runtime 写状态。
4. 运行时只在第 0 帧初始化状态；之后由冻结策略的内部有界关节 PD 和 `mj_step` 推进。
   `post_release_torque_scale=0.49` 只是统一缩放内部关节力矩，不能形成 base 外力。
5. `audit_holomotion_v24_rollout.py` 对实际状态执行独立审计，并用 MuJoCo
   `mj_geomDistance(..., distmax=0)` 查询所有 robot–chair signed distance。

对 Dramatic，`calibrate_holomotion_global_reference_bias.py` 还执行了一次受限的外环校准：
它从首次物理 rollout 的稳定全局 lag 中生成一个 controller observation reference，只在
XY 方向平滑补偿，最大总偏置 12 cm、座前最大 5 cm。**审计始终对原 physical reference
进行，不对该 controller setpoint 放宽标准。** Pianist 未使用此校准。

### 4.7 v24 已通过的静态椅子批次：Pianist 与 Dramatic

服务器的 `v24_batch25_accepted_manifest.json` 记录了两个已通过独立审计并由 SHA 绑定的
case。通过的不是“视频看着像坐下”，而是 manifest 中的 task XML、physical reference、
controller reference、actual rollout 和 audit 全部存在且哈希一致，且 audit 的
`physical_smoke_pass=true`。

通用门槛：

| 项目 | 门槛 |
| --- | ---: |
| root RMSE | ≤ 0.10 m |
| root 位置误差 P95 | ≤ 0.10 m |
| root Z 相邻帧跳变 | ≤ 0.04 m |
| 任意 robot–chair 几何穿入 | ≤ 3 mm |
| 座面法向力阈值 | ≥ 20 N 的帧计为支撑 |
| 坐下后支撑帧比例 | ≥ 60% |
| 坐下后座面载荷中位数 | ≥ 机器人重量的 25% |
| 关节限制与力矩限制 | 不得越界 |

实际结果：

| 视频 | 座面提示帧 | root RMSE | root P95 | 最大 Z 跳变 | 最小椅子距离 | 座面载荷中位数/重量 | 支撑帧比例 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pianist | 128 | 2.92 cm | 5.98 cm | 6.27 mm | -1.40 mm | 30.69% | 98.46% | 通过 |
| Dramatic | 142 | 3.74 cm | 9.25 cm | 6.49 mm | -1.83 mm | 28.08% | 88.79% | 通过 |

这里的负距离是 MuJoCo 接触合规下的毫米级软压入，仍在 3 mm 几何门槛内；它不是把 G1
运动学地插入椅面。两例均使用固定椅子、连续 `mj_step` 和内部有界 PD；manifest 显式
禁止 `per_frame_qpos_reset`、`xfrc_applied`、`mocap_weld` 与 `moving_scene_geometry`。

### 4.8 明确排除的后续候选

曾尝试把参考投影到更强的 floor-clearance 条件，试图进一步减少脚部/地面误差；在冻结
策略下它随后引入椅腿碰撞。因此该实验没有写入 accepted manifest。当前 manifest 只冻结
已独立通过审计的 artefact，避免“某个局部指标改善”覆盖新的椅腿风险。

## 5. 当前状态：哪些结论可以说，哪些不能说

可以严谨地说：

- 当前 GMR 原始 root 连续，历史跳变来源已隔离并有 SHA 合同；
- GMR 视觉脚与官方碰撞脚不存在已证实的系统性坐标错位；
- Pianist、Dramatic 在 v24 固定椅子、冻结 HoloMotion 控制和连续 `mj_step` 条件下通过了
  当前物理 smoke gate；
- 通过样例没有施加骨盆外力，没有逐帧重写机器人状态，也没有移动椅子；
- MJPC/MPC 与 SONIC 的失败结果保留为可复现证据，而不是被覆盖或包装成成功视频。

不能严谨地说：

- “MPC 已经解决了坐姿”——不正确，MPC 的 Buying 完整测试失败；
- “SONIC 已能稳定坐下”——不正确，冻结 SONIC 在该类参考上失稳；
- “三个视频都已经物理通过”——不正确，当前 accepted manifest 只有 Pianist 和 Dramatic；
- “任何椅子都能自动生成并物理可用”——不正确，语义椅子必须受 VideoMimic 证据、静态
  尺度合同、投影审计和全 robot–chair 距离门共同约束；
- “GMR 的每条动作都可无修改地成为物理轨迹”——不正确，GMR 是运动学参考，若其姿态与
  椅腿/下肢几何不相容，控制器应拒绝或保持失败证据。

## 6. 代码索引和复现顺序

### 6.1 MPC/MJPC 线

| 脚本 | 责任 |
| --- | --- |
| `scripts/scene/build_g1_mjpc_task_bundle.py` | GMR motion + 固定场景 → 无 weld/无外力的 MJPC task XML |
| `scripts/scene/materialize_mjpc_physical_motion.py` | 物化 MJPC 使用的参考运动，不作为 runtime 状态覆盖 |
| `scripts/scene/audit_g1_mpc_trajectory.py` | root、关节、力矩、座面支持和穿入联合审计 |
| `scripts/scene/audit_mjpc_chair_contact_types.py` | 按 robot link 与椅子部件拆分接触类型 |
| `scripts/scene/diagnose_mjpc_chair_contacts.py` | 定位具体碰撞 frame、geom 与接触力 |
| `scripts/scene/export_mjpc_task_keyframes_csv.py` | 导出关键帧供人工复核 |

### 6.2 SONIC 线

| 脚本 | 责任 |
| --- | --- |
| `scripts/scene/prepare_sonic_scene_compat.py` | 检查 G1/静态椅子场景对 SONIC 的兼容性 |
| `scripts/scene/prepare_sonic_physics_config.py` | 写入 scene XML、DDS domain 和关节结构配置 |
| `scripts/scene/write_sonic_playback_input.py` | 生成 SONIC 离线回放输入 |
| `scripts/scene/run_sonic_audited_sim.py` | 禁用跌倒后的隐藏重置，记录真实失败 |
| `scripts/scene/analyze_sonic_trace.py` | 分析 trace 的状态与控制事件 |

### 6.3 v24 HoloMotion 线

| 脚本/补丁 | 责任 |
| --- | --- |
| `export_gmr_motion_to_holomotion_reference.py` | GMR 29-DoF → HoloMotion `ref_*`，不修动作 |
| `build_holomotion_static_chair_scene.py` | 固定椅子 world geoms，保持策略输入 body 数不变 |
| `build_holomotion_physical_seat_reference.py` | 以真实几何选择 transition 与重力坐姿端点 |
| `select_chair_supported_seat_keyframe.py`、`transfer_verified_g1_seat_keyframe.py` | 只在椅子 seat frame 内确定性转移已验证静态姿态 |
| `run_g1_gravity_seat_landing.py` | 重力/内部关节力矩基线，检验坐姿不是悬空 |
| `calibrate_holomotion_global_reference_bias.py` | 一次受限 controller reference 校准，不替代审计 reference |
| `patches/holomotion_v24_evaluator.patch` | HoloMotion 实际椅子接触 telemetry 与受限 post-seat 控制 |
| `scripts/audit_holomotion_v24_rollout.py` | v24 独立物理 gate |
| `scripts/audit_holomotion_v24_foot_floor.py` | 双脚–地面距离 gate |
| `scripts/harden_mujoco_floor_contact.py` | 仅提高 floor material 优先级/刚度，不改变椅子或机器人参考 |
| `scripts/write_v24_accepted_batch_manifest.py`、`validate_v24_accepted_batch_manifest.py` | 冻结及复核通过 artefact 的 SHA 和标签 |
| `scripts/compose_time_synchronised_v24_2x2.py` | 只为已接受的实际 rollout 生成同步 2×2 |

## 7. 面向后续批处理的规则

新视频不得从旧目录复制 chair、motion 或视频；每条候选都应独立生成并记录：

```text
输入视频 SHA
GVHMR/SMPL motion SHA
GMR motion SHA
VideoMimic/semantic chair contract SHA
MuJoCo task XML SHA
physical reference SHA
controller reference SHA（若存在）
actual rollout SHA
独立 audit JSON SHA
```

只有在同一物理 gate 下通过的 rollout 才能进入 accepted manifest 和最终 2×2。若失败，
应保留 audit 并标为拒绝；不允许再以逐帧 root 修补、移动椅子或覆盖状态的方式“修好”。

现阶段对于 Buying，应沿同一合同重新检查其椅腿–下肢可行性与控制轨迹；不能把 Pianist
或 Dramatic 的 reference/椅子参数直接复制过去。对于真正未知场景，语义椅子仍必须来自
VideoMimic 可见几何、SMPL 接触证据和坐标/投影审计，而不是纯文本生成一把椅子。

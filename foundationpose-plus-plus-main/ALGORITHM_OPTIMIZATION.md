# FoundationPose++ 算法优化迭代文档

## 一、算法优化背景

### 1.1 FoundationPose 原生算法概述

FoundationPose（CVPR 2024 Highlight，NVIDIA）是一个基于 RGB-D 的 6D 物体位姿估计与跟踪框架。其核心由三个组件构成：

- **ScorePredictor**：评分网络，对多个位姿假设进行打分排序
- **PoseRefinePredictor（RefineNet）**：位姿精修网络，基于大核卷积神经网络，通过比较渲染深度图与观测深度图的差异，输出位姿增量修正
- **nvdiffrast**：CUDA 加速的微分光栅化渲染器

原生跟踪流程为：

```
首帧: register()
  → 随机采样旋转假设 → guess_translation() 估计初始平移 → RefineNet 迭代优化
  → ScorePredictor 评分 → 选最高分作为初始位姿

后续帧: track_one()
  → 将上一帧位姿 pose_last 作为当前帧的初始猜测
  → RefineNet 以 pose_last 为起点，迭代 5 次精修 → 输出当前帧位姿
  → pose_last = 当前帧位姿（用于下一帧）
```

**该设计存在三个结构性缺陷**：

1. **无目标检测模块**：原生 pipeline 用 SAM 直接从整张图像中分割物体并生成 mask。但 SAM 不具备语义理解能力——没有显式的位置提示（bounding box 或 point prompt）时，其对"哪个物体是目标"的判断近乎随机，经常将背景中不相关的区域误分割为目标物体，导致后续 `register()` 初始化到错误位置。

2. **无滤波机制**：`track_one()` 对 `pose_last` 完全信任。一旦 `pose_last` 被污染（遮挡、快速移动导致的错误收敛），后续所有帧的跟踪将连锁崩溃，无法自行恢复。

3. **无 2D 辅助信息**：位姿精修完全依赖 3D 深度渲染比较。在深度缺失、运动模糊、部分遮挡等场景下，RefineNet 缺乏足够信息进行正确收敛。物体的 2D 图像坐标（(cx, cy)）是一个廉价且鲁棒的信号，但原生算法未加以利用。

这些缺陷在受控实验室条件下（YCB 数据集、缓慢移动）不易暴露，但在真实操作场景（手持物体快速移动、遮挡、出画再入画）中会导致跟踪频繁失败。

### 1.2 优化目标

构建一套在真实操作场景下稳定运行的 6D 位姿跟踪系统，具体目标包括：

- 物体识别准确率显著提升
- 静态/慢速/遮挡/高动态等场景下均稳定跟踪
- 遮挡后快速恢复
- 物体出画后入画可恢复
- 位姿无高频抖动

---

## 二、优化迭代总览

| 迭代 | 优化项 | 解决的问题 |
|------|--------|-----------|
| 1 | Qwen3-VL 检测 + SAM-HQ 分割 | 物体识别错误率高 |
| 2 | 12D 卡尔曼滤波器 | 位姿无滤波、逐帧噪声累积 |
| 3 | Cutie 2D 跟踪器 | 缺少 2D 视觉辅助信息 |
| 4 | 遮挡检测与恢复 | 遮挡后跟踪完全失效 |
| 5 | 高动态响应 | 快动/弹飞/出画场景跟踪失败 |

---

## 三、优化项详述

### 3.1 优化 1：引入 VLM 检测 + SAM-HQ 分割

#### 3.1.1 问题现象

SAM（Segment Anything Model）能对任意物体进行高质量分割，但其自身不具备语义识别能力。原生 pipeline 用 SAM 直接从整张图像中分割物体并生成 mask，不经过任何目标检测步骤。然而在实际测试中发现，SAM 在无显式位置提示的情况下无法区分"哪个物体是目标"：

- 即使目标物体完整且清晰地出现在图像中心，SAM 仍可能将图像边缘的不相关区域误分割为目标物体
- 分割结果高度不稳定，同一场景连续两次运行可能输出完全不同的 mask
- 当图像中存在多个物体时，SAM 无法根据语义信息（如"鼠标"）选择正确的目标
- 分割错误直接导致 FoundationPose register 拿到错误的 mask → 初始化到错误物体 → 整个跟踪 pipeline 从第一帧即失效

#### 3.1.2 原因分析

SAM 的设计理念是"通用分割"而非"智能识别"——它接受位置提示（bbox/point），输出高质量 mask，但**不决定提示点从何而来**。原生 pipeline 让 SAM 在无位置提示的情况下直接处理整张图像，分割结果由 SAM 内部的通用性启发式决定——在图像中搜索"所有可能的物体"并输出其中某一个，不保证该物体是用户期望的目标。要解决这一问题，必须引入一个具备语义理解能力的前端模块来提供物体位置。

#### 3.1.3 优化方案

引入 **Qwen3-VL-30B** 作为检测前端，通过 OpenAI 兼容 API 托管为远程服务（192.168.10.242:12067），与本地 SAM-HQ 组成"VLM 识别 → SAM 分割"两级流水线。这是整个 pipeline 中新增的一个独立模块——在原生 SAM 之前插入 VLM 检测步骤，用语义信息引导分割方向。

```
Qwen3-VL-30B（检测）          SAM-HQ（分割）
"图片中的鼠标在哪？"          "剪出鼠标的mask"
        ↓                           ↓
    bbox (x,y,w,h)              精细mask
        ↓                           ↓
        └────── 组合 ──────→  RGBA抠图 + FoundationPose初始化
```

**作用**：30B 参数规模的 VLM 具有强大的零样本图像理解能力——无需微调，直接根据自然语言描述定位任意物体。原生 SAM 本身不含语义信息，无法根据"鼠标"这类文本描述确定目标；传统目标检测器（如 YOLO）需要为每类物体专门训练且泛化能力差。VLM 兼顾了"开集检测"（任意物体均可通过文本描述定位）与"高精度定位"，弥补了 SAM 在语义理解上的先天缺陷。

**方案设计**：

```
Qwen3-VL-30B API (远程)
  ↓ POST /v1/chat/completions (OpenAI 兼容协议)
  ↓ 输入: 物体中文描述 + 图像 base64
  ↓ 输出: JSON [{bbox_2d: [x1, y1, x2, y2], label: "..."}] (千分比坐标)
  ↓
obj_bbox.py: 解析 bbox → (x, y, w, h) 像素坐标
  ↓
SAM-HQ API (本地 :9002): bbox → 精细 mask → 0_mask.png + 0_mask_rgba.png
```

#### 3.1.4 算法流程

```
Step 1: 构建 Prompts
  - 系统提示: 物体中文描述（如"鼠标"）
  - 用户提示: "你能在图片里找到{物体名}的锚框吗？"
  - 图像: 首帧 RGB base64 编码

Step 2: 调用 Qwen3-VL-30B API
  - Endpoint: POST /v1/chat/completions
  - Response: choices[0].message.content

Step 3: 解析输出
  - 格式 1: JSON → bbox_2d 千分比坐标 → × img_W/1000 → 像素
  - 格式 2: 文本 (x1, y1, x2, y2) → 千分比 → 像素
  - 格式 3: 文本 [x1, y1, x2, y2] → 像素
  - 自动去除 markdown 代码块包裹

Step 4: SAM-HQ 分割
  - BBox → mask (0=背景, 255=物体)
  - mask × RGB → RGBA 透明背景抠图
```

#### 3.1.5 效果

将物体检测从前端 pipeline 中自动化，无需人工标注或预设规则。Qwen3-VL-30B 的零样本检测能力使系统可以直接根据物体名称（中文描述）自动定位目标，识别准确率接近 100%，显著优于任何基于固定规则或传统检测器的方法。

---

### 3.2 优化 2：12D 卡尔曼滤波器

#### 3.2.1 问题现象

原生 FoundationPose 将每一帧 `track_one()` 的结果直接作为下一帧的初始位姿。在深度噪声、运动模糊或部分遮挡的情况下，`track_one()` 的 RefineNet 输出会包含随机误差。由于没有任何滤波，这些逐帧误差直接累积到下一帧的初始猜测中→ 初始猜测偏离真值 → RefineNet 收敛到局部最优而非全局最优 → 位姿漂移 → 连锁恶化。

具体表现为：(1) 物体静止时，3D BBox 和坐标轴存在持续的高频微抖动；(2) 连续跟踪时间越长，位姿中心越偏离物体实际位置。

#### 3.2.2 原因分析

从信号处理角度看，FoundationPose 的逐帧位姿构成了一个开环系统：

```
pose_{t-1} → track_one() → pose_t → 作为 pose_{t+1} 的初始猜测
                ↑
         测量噪声直接注入下一帧, 无任何衰减
```

每个`track_one()`都等价于一个非线性最小二乘优化器，其输出等于"真值 + 测量噪声 + 优化残差"。在开环设计中，测量噪声和优化残差不经任何衰减直接馈入下一帧。从控制论角度看，这是一个无反馈控制器的开环系统——输出直接等于测量，没有状态估计来消除噪声。

#### 3.2.3 优化方案

引入 12 维卡尔曼滤波器（KalmanFilter6D），将位姿跟踪转化为**状态估计问题**。

**状态空间定义**：

```
x = [tx, ty, tz, rx, ry, rz, v_tx, v_ty, v_tz, v_rx, v_ry, v_rz]ᵀ
```

- 前 6 维：物体在相机坐标系中的 6D 位姿（平移 × 3 + 欧拉角旋转 × 3）
- 后 6 维：对应的 6D 速度（线性速度 × 3 + 角速度 × 3）

**状态转移（预测步）**：

```
x_{t|t-1} = F × x_{t-1|t-1}
F = [I₆  Δt·I₆ ]
    [0     I₆  ]
其中 Δt = 1（帧间隔归一化）

P_{t|t-1} = F × P_{t-1|t-1} × Fᵀ + Q

Q = diag([
    σ²_pos_trans × |tz|² × 3,   # 平移位置过程噪声
    σ²_pos_rot   × |rx|² × 3,   # 旋转过程噪声
    σ²_vel_trans × |tz|² × 3,   # 平移速度过程噪声
    σ²_vel_rot   × |rx|² × 3,   # 旋转速度过程噪声
])
```

**观测更新（修正步）**：

KF 接收来自 `track_one()` 的 6D 完整位姿作为观测：

```
y = H × x  (H = [I₆ | 0₆])
S = H × P × Hᵀ + R
K = P × Hᵀ × S⁻¹  (Kalman Gain)
x = x + K × (z - y)
P = P - K × H × P
```

**噪声参数设计**：

| 参数名 | 物理含义 | 默认值 |
|--------|---------|--------|
| `_std_weight_trans` | track_one 平移测量不确定性 | 1/10 |
| `_std_weight_rot` | track_one 旋转测量不确定性 | 1/20 |
| `_std_weight_vel_trans` | 平移速度过程噪声 | 1/10 |
| `_std_weight_vel_rot` | 旋转速度过程噪声 | 1/40 |

所有噪声均按当前状态的 `|tz|` 和 `|rx|` 缩放——物体越远，测量不确定性越大。

**作用**：KF 将 FoundationPose 的输出从"直接使用"变为"测量源之一"。滤波器根据噪声参数自动权衡"该信测量多少"。噪声大的帧（如深度不足、运动模糊）自动获得低权重；噪声小的帧自动获得高权重。这从根本上切断了开环系统中的噪声累积链。

#### 3.2.4 算法流程

```
首帧:
  measurement = get_6d_pose_arr_from_mat(register_result)
  kf_mean, kf_cov = kf.initiate(measurement)
  → kf_mean = [tx,ty,tz,rx,ry,rz, 0,0,0,0,0,0]
  → kf_cov = diag([位置: (0.2×σ_trans×scale)², 速度: (1.0×σ_vel×scale)²])

每帧末尾 (predict, 为下一帧准备):
  x_{t+1|t} = F × x_t           (position += velocity × Δt)
  P_{t+1|t} = F × P_t × Fᵀ + Q  (协方差累积过程噪声)
  → 输出: 下一帧 track_one 的初始位姿猜测

当前帧收到 track_one 后 (update):
  z = get_6d_pose_arr_from_mat(pose)  (6维观测)
  y = H × x                           (6维投影, H=[I₆|0₆])
  S = H × P × Hᵀ + R                  (创新协方差)
  K = P × Hᵀ × S⁻¹                     (Kalman Gain, 6×12)
  x = x + K × (z - y)                 (状态修正)
  P = P - K × H × P                   (协方差缩减)
```

注意 predict 和 update 在不同时间执行：predict 在上一帧末尾推算出下一帧的初始状态，update 在当前帧收到观测后进行修正。两者交替构成了 KF 的"预测-修正"闭环。

#### 3.2.5 效果

- 静态场景下坐标轴抖动基本消除，帧间位姿变化更平滑
- 连续跟踪稳定性显著提高，不会"跑着跑着就偏了"

---

### 3.3 优化 3：Cutie 2D 跟踪器辅助

#### 3.3.1 问题现象

仅使用 KF 对 FoundationPose 的 6D 结果进行滤波后，静态/慢速场景效果较好，但在物体快速平移时（如在桌面上快速滑动鼠标），位姿中心落后于物体实际位置。

#### 3.3.2 原因分析

KF 的预测步基于**匀速运动假设**（`x_t = x_{t-1} + v × Δt`）。当物体突然加速时，KF 的速度分量仍为加速度前的值，预测位置落后于实际位置。KF 需要 3-5 帧连续观测才能将速度估计"追上来"。这 3-5 帧的滞后在高动态场景下表现为明显的"跟丢感"。

更深层的问题是：6D 位姿是 RefineNet 通过比较渲染深度和观测深度迭代优化得出的——这是一个计算密集型过程，天然"慢半拍"。相比之下，物体在 2D 图像上的位置 (`cx, cy`) 可以通过视觉跟踪器直接在 RGB 图像上实时获取，响应几乎无延迟。FoundationPose 原生算法没有利用这一 2D 信号。

#### 3.3.3 优化方案

引入 Cutie 作为 2D 视频分割跟踪器，提供帧级物体 2D 中心位置。Cutie 输出的 bounding box 中心 `(cx, cy)` 作为 KF 的第二个独立观测源。

**选用 Cutie 的原因**：Cutie 是一个基于记忆的视频分割模型，能够在首帧得到精确 mask 后，在后续帧中通过特征匹配持续输出精确分割。相比传统光流或特征点跟踪，Cutie 在形变、旋转、部分遮挡下的鲁棒性更好。

**架构**：

```
Cutie.track(rgb) → bbox_2d → (cx, cy)
     ↓
adjust_pose_to_image_point(est.pose_last, cx, cy)
     ↓
tx_new = (cx - K_cx) × tz / fx      ← 从像素坐标反算物理坐标
ty_new = (cy - K_cy) × tz / fy
     ↓
track_one(rgb, depth, 修正后的位姿)
     ↓
KF.update(track_one 6D)              ← 测量源 1
KF.update_from_xy(cx, cy 反算的 xy)  ← 测量源 2
```

**Cutie xy 测量噪声设计**：

```
R_xy = (σ_xy_trans × scale_xy)² × I₂

scale_xy = max(‖[tx, ty]‖, |tz| × 0.2)
```

- `‖[tx, ty]‖`：物体偏离光轴越远，Cutie 噪声应越大（透视效应放大像素误差）
- `|tz| × 0.2`：保护下限——物体在图像中心时，噪声不会被低估为 0（否则 KF 过度信任 Cutie → 抖动）。0.2 对应等效 ~3px 的 Cutie 噪声容忍度

#### 3.3.4 算法流程

```
首帧:
  Cutie.initialize(rgb_0, mask=SAM_mask_0)
    → cutie_processor.step(frame, mask)  传播首帧特征到 memory bank
    → bbox_0 = mask 的包围矩形
    → 后续帧初始状态就绪

每帧:
  Step 1: Cutie.track(rgb_t)
    → cutie_processor.step(frame)         基于 memory bank 匹配当前帧
    → output_prob_to_mask                 概率 → 二值 mask
    → cv2.erode(mask, kernel=5)           腐蚀去噪
    → bbox_2d = mask 的包围矩形 (x, y, w, h)

  Step 2: 像素中心 → 物理坐标
    cx, cy = bbox_2d 中心
    adjust_pose_to_image_point(est.pose_last, cx, cy)
      → tx_new = (cx - K_cx) × tz / fx    2D→3D 反投影
      → ty_new = (cy - K_cy) × tz / fy
      → 更新 est.pose_last 的 tx, ty，旋转不变

  Step 3: est.pose_last → track_one 初始位姿

  Step 4: KF.update(track_one 结果)        ← 测量源 1 (3D)
          KF.update_from_xy(cx,cy 反算)    ← 测量源 2 (2D)
          两路独立测量同时注入，KF 按各自噪声权重自动融合
```

#### 3.3.5 效果

- 物体平移跟踪的实时性明显提升，Cutie 提供的快速 2D 信号有效弥补了 track_one 的滞后
- 在正常操作场景下，两路测量源互补——track_one 提供精确但慢速的 3D 位姿，Cutie 提供快速但 2D 的位置线索——KF 自动按噪声权重融合，消除了单源滤波的滞后问题

---

### 3.4 优化 4：遮挡鲁棒性

#### 3.4.1 问题背景

在真实操作场景中，物体频繁被手部分或完全遮挡。原生 FoundationPose 对遮挡完全无处理——当手遮住物体时，深度图中物体表面的深度测量被手的深度替代，RefineNet 在错误深度上迭代优化 → 位姿被拉到手上 → 手移开后位姿停留在错误位置 → 永远无法恢复。

这个问题在优化 2-3 后仍然存在——KF 和 Cutie 均无法区分"物体被遮挡"和"物体正常可见"两种状态。

#### 3.4.2 遮挡检测

**原理**：当物体被手部分遮挡时，Cutie 输出的 bbox 中心 `(cx, cy)` 可能仍位于物体可见部分，但深度图中该位置的深度值会从"物体表面深度"变为"手表面深度"（手比物体更靠近相机）。通过在 bbox 中心区域采样深度，可以判断中心是否被遮挡：

**实现**：

```
Step 1: 在 (cx, cy) 处取深度采样窗口
        hw = max(2px, bbox_width × 0.15)
        hh = max(2px, bbox_height × 0.15)
        patch = depth[y0:y1, x0:x1]

Step 2: 统计有效深度点数
        valid_count = count(patch > 0.001 && isfinite)

Step 3: 判定遮挡
        is_occluded = bbox_valid && (valid_count < 5)
```

选择中心区域采样而非全图的原因：边缘像素可能仍属于物体（露出了一截），但中心被遮 → 物体的表面深度丢失 → RefineNet 无法正确优化 → 需要判定为遮挡状态。

#### 3.4.3 遮挡下的 KF 保护

当检测到遮挡时，KF **跳过 `update()`，只执行 `predict()`**：

```
遮挡帧: KF.predict()  → x_{t+1} = F × x_t  (仅运动模型外推)
正常帧: KF.update()   → 正常融合测量

关键设计: 遮挡期间不进任何测量 → KF 状态不受污染
```

**为什么不用 `register()` 直接重建**：register() 需要可靠 mask，但遮挡时 SAM 无法提供 clean mask；用投影 bbox 构造的矩形 mask 质量差，register() 的位姿精度不足以继续跟踪。KF 外推在短期（≤10 帧）内精度可接受。

#### 3.4.4 Cutie 记忆污染与恢复

遮挡期间 Cutie 内部记忆只看到物体的局部外观。手移开后物体完全露出，但 Cutie 记忆仍是"被遮掉一块"的残缺特征——bbox 输出持续缩小（ratio < 0.5）——需要 ~200 帧才能自行恢复。

**恢复策略（CutieReset）**：

```
触发条件:
  (1) 连续 ≥2 帧丢失后当前帧 bbox 有效且有深度
  (2) 物体可见但 Cutie bbox 面积 < projected bbox 面积 × 0.5

执行:
  → Cutie.initialize(current_rgb, mask=projected_bbox)
  → 清除被污染的记忆，用投影 bbox 重建初始 mask
  → 不调用 register()：位姿从 KF 继承，由 track_one refine 修正
  → was_occluded = True → 触发下方 tz 漂移修正
```

**注意**：re-register 时物体可能仍处于部分遮挡状态——用残缺 mask 调用 `register()` 会产生错误位姿。因此不重置 3D 位姿，只用 KF 预测 + depth 中线深度修正 tz。

#### 3.4.5 恢复帧 tz 漂移修正

遮挡期间 KF 的 tz 分量仅靠速度外推。手移开后 tz 可能已偏移。恢复帧重新从深度图读取物体中心处的深度中位数，直接修正：

```
if is_recovering && tz_from_depth is not None:
    est.pose_last[2, 3] = tz_from_depth    ← 修正 track_one 起点的深度
    kf_mean[2] = tz_from_depth            ← 修正 KF 状态中的深度
```

注意 `kf_mean[2]` 的修正发生在 `kf.update(pose)` **之后**、`kf.update_from_xy(xy)` **之前**。这样 `measurement_xy` 的反算（`tx = (cx - K_cx) × tz / fx`）用的是修正后的正确 tz——避免用漂移的 tz 计算出错误的 tx/ty 注入 KF。

#### 3.4.6 物体出画后入画恢复

物体完全出画 10+ 帧后，KF 如果持续 predict → 位置带速度越漂越远 → KF 状态飞到画面外 → `project_real_dims_bbox_to_2d()` 将 3D bbox 投影到画面外 → `projected_bbox = None` → CutieReset 永远不会触发 → **完全无法恢复**。

**修复**：出画超过 10 帧时清零速度、冻结位置：

```
if consecutive_bad_frames > 10:
    kf_mean[6:12] = 0.0   # 清零速度分量
    # 不调用 kf.predict() → 位置保持在最后有效位姿附近
```

这样物体回来后，`projected_bbox` 仍在画面内有效位置 → CutieReset 可以正常触发。

#### 3.4.7 恢复后抖动消除：KF 协方差重置

**问题现象**

物体重新入画后，虽然 CutieReset 能在 1-2 帧内恢复 2D BBox，但 3D BBox 和坐标轴在恢复后前 5-10 帧出现明显高频抖动，幅度逐渐收敛至正常水平。

**原因分析**

遮挡期间 KF 仅执行 predict。每帧 predict 向协方差矩阵 P 累积过程噪声 Q：

```
P_{t} = F × P_{t-1} × Fᵀ + Q
```

其中 Q 包含速度和位置的不确定性项。遮挡 ≥10 帧后 P 已膨胀至正常值的 10 倍以上——KF 变得"极不确定自己的估计"。

物体重新入画时，CutieReset 触发 → Cutie 用投影 bbox 矩形 mask 冷启动（边界精度不如 SAM-HQ）→ bbox 中心噪声比正常大 2-3 倍。同时 track_one 从冻结位置出发，首帧收敛也不够精确。

膨胀的 P 导致 Kalman Gain 过高：

```
K = P × Hᵀ × (H × P × Hᵀ + R)⁻¹
  ↑ P 大 → K 大 → 测量被几乎全采信
```

两路带噪声的测量（Cutie 冷启动 + track_one 残差）被大 K 放大注入 KF 状态 → 位姿高频抖动。需要 5-10 帧正常 update 才能将 P 重新压回正常水平——抖动恰好对应这个收敛期。

**优化方案**

CutieReset 触发时，不从膨胀的 P 继续 update——而是调用 `kf.initiate(track_one 结果)` 直接重建 KF：

```
kf_needs_reset = True (在 CutieReset 块中设置)

Step 6 执行时:
  if kf_needs_reset:
      kf_mean, kf_cov = kf.initiate(get_6d_pose_arr_from_mat(pose))
      if tz_from_depth: kf_mean[2] = tz_from_depth
      kf_needs_reset = False
```

`initiate` 将 P 直接恢复为初始对角阵（位置分量不确定性较大、速度分量从零开始），Kalman Gain 立即回归正常大小。后续帧走正常的 `kf.update + kf.update_from_xy` 双源融合。

**为什么不用 `update` 慢慢收敛**：`update` 沿对角线方向逐步缩减 P 中对应分量的方差，收敛速度由 innovation 和当前 P 的比值决定。对已膨胀至 10 倍以上的 P，5-10 帧才能压回正常范围——这恰是用户感知到的抖动期。`initiate` 将 P 直接回到起点，跳过收敛期。

**效果**

恢复首帧即恢复平滑跟踪，无明显抖动收敛过程。结合位置冻结（3.4.6），完整恢复链：入画 1 帧 CutieReset + KF 重建 → 第 2 帧开始平稳跟踪。

#### 3.4.8 算法流程（遮挡处理完整逻辑）

```
每帧：
  ├─ Cutie.track(rgb) → bbox_2d
  ├─ 投影 pred_pose → projected_bbox
  ├─ 中心定位 + 遮挡检测(is_occluded)
  │    ├─ Cutie bbox > projected bbox × 1.6 → 模糊帧 → 用投影中心
  │    └─ 正常 → 用 Cutie 中心
  ├─ cutie_corrupted 检测 (bbox < 投影 50%)
  ├─ CutieReset?
  │    ├─ 条件: need_cutie_reset || cutie_corrupted
  │    └─ 执行: Cutie.initialize(投影bbox mask) + 协方差重置标记
  │
  ├─ KF predict (遮挡/丢失) 或 冻结 (>10帧)
  ├─ adjust_pose_to_image_point(cx, cy)  # 逐帧初始位姿修正
  ├─ track_one(rgb, depth)  → pose
  └─ KF 双源融合
       ├─ kf.update(track_one)     → 测量源 1
       ├─ kf.update_from_xy(cutie) → 测量源 2
       └─ 恢复帧 → kf_mean[2] = tz_from_depth
```

#### 3.4.9 效果

- 短暂遮挡（手快速经过）：跟踪稳定，位姿不漂移
- 长时间遮挡后恢复：CutieReset 在 1-2 帧内恢复正确 bbox
- 出画后入画：位置冻结 + 协方差重置 → 可恢复且恢复后平滑
- 遮挡期间位姿不会被手的深度测量污染

---

### 3.5 优化 5：高动态场景鲁棒性

#### 3.5.1 问题现象

手拿着物体快速移动、弹飞物体、上抛下落等场景下：

1. **位姿滞后**：bbox 和坐标轴落在物体后面，3-5 帧后才追上
2. **弹飞后跟丢**：手指弹飞物体的瞬间，位姿停在弹飞位置，物体已飞出 bbox 范围，彻底丢失
3. **上抛时 bbox 无法感知近大远小**：物体远离相机时 bbox 仍然很小，无法正确包围物体
4. **物体出画后入画恢复时抖动**：通过出画位置冻结 + KF 协方差重置解决

#### 3.5.2 原因分析

问题 1-2 的根因涉及多个层面：

**层面 1：KF 惯性**

KF 的预测步基于匀速模型（`x_{t+1} = x_t + v × Δt`）。弹飞瞬间物体获得巨大加速度，但 KF 的速度估计仍为加速度前的低速值。KF 预测位置 = 实际位置 - 加速度缺口 → 落后于物体。

**层面 2：track_one 的收敛域限制**

`RefineNet.predict()` 的核心设计：

```python
trans_delta = tanh(output["trans"]) × trans_normalizer
# tanh 函数将平移增量限制在 [-normalizer, +normalizer] 范围内
# trans_normalizer ≈ 0.02m，即每次迭代最多修正 2cm
```

迭代 5 次，总计最多修正约 10cm。弹飞/快移时，两帧之间物体位移可能达到 20-50cm——远超 track_one 的收敛域。KF 预测位置（旧位置 + 低速外推）作为 `track_one` 的起点，和实际位置差距过大 → RefineNet 收敛到局部最优或完全失败。

**层面 3：KF 速度分量污染**

弹飞瞬间 KF 速度分量被加速度前的低速值占据。`kf.predict()` 将错误的速度加到位置上。在连续高动态场景中，速度分量持续被历史加速度污染 → 在完全静止前的若干帧内持续产生"惯性尾迹"。

#### 3.5.3 优化方案：三种策略

**策略 A：运动检测 + 自适应 KF 噪声**

```
每帧计算 Cutie bbox 中心帧间位移（像素）:
  disp = sqrt((cx - prev_cx)² + (cy - prev_cy)²)
  motion_level = 0.4 × motion_level + 0.6 × disp  (EMA 平滑)

判定:
  is_high_motion = (motion_level > 12 px/frame)

自适应:
  if is_high_motion:
      kf.set_process_noise_scale(3.0)
      kf_mean[6:12] = 0.0  (清零速度, predict 前)
  else:
      kf.set_process_noise_scale(1.0)
      (正常 predict)
```

**策略 B：track_one 初始位姿直接跳转**

不管 KF 预测是否准确，直接将 track_one 的初始 (tx, ty) 修正到当前帧 Cutie bbox 中心对应的物理位置：

```
每帧: est.pose_last = adjust_pose_to_image_point(cx, cy)
  → tx = (cx - K_cx) × tz / fx
  → ty = (cy - K_cy) × tz / fy
  → 旋转分量保持不变
```

这个设计的精妙之处在于：(tx, ty) 由像素中心直接反算，**与 KF 预测无关**。即使 KF 速度完全错误、预测位置差了几十厘米，`adjust_pose_to_image_point` 也会把初始位置直接拉到正确 bbox 中心。track_one 从正确位置出发 → 收敛域不再是瓶颈。

**对正常帧的影响**：Cutie 中心 1-3px 的 jitter 对应物理世界 1-2mm → track_one 秒级 refine → KF 滤波 → 噪声不可见。弹飞时 Cutie 中心跳了 100px → tx,ty 直接跳 20cm → track_one 从接近真值的位置 refine → 收敛成功。

**策略 C：KF 速度清零**

在高动态模式下，predict 前将速度分量清零：

```
if is_high_motion:
    kf_mean[6:12] = 0.0   # 速度清零
kf_mean = kf.predict(kf_mean, P)  # predict 实际退化为 x += 0
```

这等价于"放弃速度外推，完全相信每帧的测量"——因为加速度不可预测，速度估计不可靠。清零后在 predict 中退化为恒等映射（位置不变），测量更新由策略 A 中放大的噪声参数给予更大权重。

#### 3.5.4 高动态 vs 低动态：两种策略的切换

| 状态 | 条件 | KF 策略 | 速度处理 | 效果 |
|------|------|---------|---------|------|
| 低动态 | `motion_level ≤ 12` | `process_noise_scale=1.0` | 正常 predict(v×Δt) | 平滑匀速跟踪 |
| 高动态 | `motion_level > 12` | `process_noise_scale=3.0` | 清零后再 predict | 快速跟随、无惯性 |

切换由 EMA 平滑的帧间位移控制。EMA 的 `MOTION_DECAY=0.4` 确保不会因单帧噪声触发误切。

#### 3.5.5 协方差膨胀抑制：高动态退出时的平滑恢复

**问题现象**

高动态操作（弹飞、快速平移）结束后物体恢复平稳运动，但切换回低动态模式后的前几帧，3D BBox 和坐标轴出现可感知的微抖动。该现象在物体先出画、后入画的高动态场景中尤为明显。

**原因分析**

协方差矩阵 P 的膨胀速度与速度不确定性呈正比。高动态模式下，策略 A 将速度噪声放大 3 倍（`process_noise_scale=3.0`），策略 C 清零速度。两个操作叠加的效果是：

```
高动态每帧 predict:
  P += FPFᵀ + Q(scale=3.0)      ← Q 放大 3 倍 → P 膨胀速度 3 倍
  kf_mean[6:12] = 0 后再 predict → 位置不变，但 P 照常累积

正常每帧 predict:
  P += FPFᵀ + Q(scale=1.0)      ← Q 正常

高动态持续 N 帧:
  P 膨胀 ≈ 正常 N 帧的 3 倍     ← 原因是 scale 直接放大速度过程噪声 Q
```

当高动态结束时（`motion_level` 回落至 ≤12），模式切回低动态，KF 恢复 `update`。此时 P 已被高动态期间的放大 Q 显著推高 → Kalman Gain 过大 → 前几帧的测量（track_one + Cutie）被过度采信 → 退出高动态时出现短期抖动。

对于先出画、后入画的场景，P 膨胀更严重：出画期间长时间 predict 累积 + 入画瞬间高动态触发 3× 放大 → P 可膨胀至正常值的 10 倍以上。

**优化方案**

Code>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Edit">
<｜｜DSML｜｜parameter name="new_string" string="true">#### 3.5.5 协方差膨胀抑制：高动态退出时的平滑恢复

**问题现象**

高动态操作（弹飞、快速平移）结束后物体恢复平稳运动，但切换回低动态模式后的前几帧，3D BBox 和坐标轴出现可感知的微抖动。该现象在物体先出画、后入画的高动态场景中尤为明显。

**原因分析**

协方差矩阵 P 的膨胀速度与过程噪声 Q 呈正比。高动态模式下，策略 A 将速度噪声放大 3 倍（`process_noise_scale=3.0`），策略 C 清零速度。两个操作叠加：

```
高动态每帧 predict:
  P += FPFᵀ + Q(scale=3.0)      ← Q 放大 3 倍 → P 膨胀速度 3 倍
  kf_mean[6:12] = 0 → predict  → 位置不变，但 P 照常累积

低动态每帧 predict:
  P += FPFᵀ + Q(scale=1.0)      ← Q 正常

高动态持续 N 帧:
  P 膨胀 ≈ 正常 N 帧的 3 倍
```

当高动态结束时（`motion_level` 回落至 ≤12），模式切回低动态。此时 P 已显著膨胀 → Kalman Gain 过大 → 前几帧的测量被过度采信 → 退出高动态时出现短期抖动。5-10 帧后 P 通过正常 update 逐步缩减至正常水平 → 抖动自行收敛。

对于先出画、后入画的场景，膨胀叠加更严重：出画期间长时间 predict 累积 + 入画瞬间高动态触发 3× 放大 → P 膨胀至正常值的 10 倍以上。

**优化方案**

不等待 P 自然收敛（5-10 帧），而是在 CutieReset 触发后直接调用 `kf.initiate(track_one 当前结果)` 重建 KF——P 恢复初始对角阵，Kalman Gain 立即回归正常大小：

```
标志设置 (CutieReset 块中):
  kf_cov_needs_reset = True

Step 6 执行时:
  if kf_cov_needs_reset:
      kf_mean, kf_cov = kf.initiate(get_6d_pose_arr_from_mat(pose))
      if tz_from_depth: kf_mean[2] = tz_from_depth
      kf_cov_needs_reset = False
```

`initiate` 将 P 重建为初始对角阵（位置方差适中、速度从零开始）。后续帧走正常 `kf.update + kf.update_from_xy` 双源融合，不再有膨胀的 P 导致的过冲。

**与策略 C 的关系**：策略 C（速度清零）操作 `kf_mean`（状态均值），抑制高动态期间的位置惯性；协方差重置操作 `kf_cov`（状态不确定性），消除退出高动态时的测量过冲。两者分别解决不同维度的问题，互不冲突。

**效果**

高动态结束后——无论是自然减速退出还是出画后入画恢复——位姿立即平稳，无因 P 膨胀导致的短期抖动。

#### 3.5.6 模糊帧污染检测

高速运动导致运动模糊 → SAM 在模糊帧上将手和物体分割为一个整体 → Cutie bbox 异常大（包含手）→ 中心偏到手上。

**检测与修复**：

```
if Cutie bbox 面积 > projected bbox 面积 × 1.6:
    (cx, cy) = projected_bbox 中心  ← 用 KF 预测位姿投影的真实尺寸中心
    深度采样窗口 = projected_bbox 尺寸
```

投影 bbox 基于真实 3D 物理尺寸 + KF 位姿投影到 2D，不受图像模糊影响。模糊帧的 Cutie bbox 比投影大 30%+ → 不可能只是物体变大 → 判定为模糊污染。

#### 3.5.7 效果

- 物体上抛/快速平移：位姿中心跟上，无明显滞后
- 弹飞物体：不跟丢，位姿持续跟踪
- 模糊帧：中心不被手的位置污染
- 物体出画后入画：KF 协方差重建 → 恢复首帧即平稳无抖动
- 高/低动态自动切换，无人工介入

---

## 四、当前待解决问题

### 4.1 深度快速变化（上抛下落）场景下 bbox 缩放不足

**现象**：物体上抛后再下落时，位姿中心跟随尚可，但 3D bbox 的 2D 投影无法正确体现近大远小——物体远离时 bbox 仍然偏小。

**分析方向**：可能与 KF prediction 中的 tz 分量修正时机有关。`kf_tz` 在 track_one 之前读取，而 track_one 后 KF update 才更新 tz——measurement_xy 的反算可能使用了一帧前的过期 tz。该问题仍在调试中。

---

## 五、核心文件清单

| 文件 | 功能 | 优化涉及 |
|------|------|---------|
| `src/obj_pose_track.py` | 6D 位姿跟踪主流程 | 全部优化 |
| `src/utils/kalman_filter_6d.py` | 12D KF 滤波器 | 优化 2、3、5 |
| `src/VOT.py` | Cutie 2D 跟踪器封装 | 优化 3 |
| `src/utils/obj_bbox.py` | Qwen3-VL BBox 请求 | 优化 1 |
| `src/WebAPI/hq_sam_api.py` | SAM-HQ 分割 API | 优化 1 |
| `scripts/run_pipeline.py` | Pipeline 编排 | 整体流程 |
| `scripts/hunyuan3d_bridge.py` | Hunyuan3D Bridge | Mesh 生成 |

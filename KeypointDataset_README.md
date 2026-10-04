# 关键点数据集录制说明（get_dateset_keypoint.py）

本目录原有 `get_dateset.py`（自动生成带姿态信息的 YOLO 检测数据集）**未做任何修改**。
新增的 `get_dateset_keypoint.py` 是它的副本 + **关键点(keypoint)录制**功能，用于后续训练
关键点检测 / 位姿(SO3)估计神经网络。

## 1. 文件清单

| 文件 | 作用 |
| --- | --- |
| `get_dateset_keypoint.py` | 录制脚本本体：取图 + YOLO 检测框 + 9 个关键点标签 + COCO 关键点 JSON + 训练 yaml |
| `_verify_keypoints.py` | 离线自检：几何/投影/标签格式/可见性/JSON/yaml 共 39 项，不需要 UE4 |
| `_verify_smoke.py` | 离线端到端冒烟测试：用假接口跑 `main()`，真的写 8 张图并逐项校验落盘内容 |
| `get_dateset.py` | 原始脚本（只读基线，自检脚本会用 sha256 校验它没被改动） |
| `Config.json` | 传感器与人参配置；其中的 `dataset` 段可配置保存目录、`MinKeypoints` 等 |

## 2. 运行

```bat
:: 与原始脚本一致：先启动 RflySim3D，脚本自己会切到 Grasslands 地图
python get_dateset_keypoint.py

:: 常用参数
python get_dateset_keypoint.py --save-dir D:/kp_dataset --no-timestamp
python get_dateset_keypoint.py --min-keypoints 6          :: 只保留可见关键点>=6 的样本
python get_dateset_keypoint.py --no-coco                  :: 只写 YOLO-Pose 标签
python get_dateset_keypoint.py --print-keypoints          :: 终端打印每个关键点像素坐标
python get_dateset_keypoint.py --headless --max-samples 2000   :: 无人值守录 2000 张
```

运行中把焦点放在 OpenCV 窗口上按 `q` 退出；退出（含 Ctrl+C）时会自动把 `keypoints.json`
落盘并打印统计。预览窗口 `Keypoints` 会把 9 个点、立体框、检测框画在图上，便于确认标定正确。

离线自检（不需要 RflySim，不会产生数据集目录）：

```bat
python _verify_keypoints.py     :: 39 项几何/格式自检，输出 _verify_keypoints.png
python _verify_smoke.py         :: 端到端写盘冒烟测试（临时目录，跑完自动删除）
```

## 3. 输出目录结构

```
<SaveDir>/[时间戳]/
├── images/            0000.jpg ...            原始图像（干净，不带任何绘制）
├── labels/            0000.txt ...            YOLO-Pose 关键点标签
├── keypoints.json                             COCO-Keypoints 标签（含 3D 坐标与目标位姿）
└── dataset_pose.yaml                          Ultralytics YOLO-Pose 训练配置模板
```

## 4. 标签格式

### 4.1 YOLO-Pose（`labels/*.txt`，每行一个目标）

```
<class> <cx> <cy> <w> <h> <k0x> <k0y> <k0v> ... <k8x> <k8y> <k8v>
```

共 `5 + 3×9 = 32` 个字段，全部为浮点数（关键点 x/y 归一化到 `[0,1]`）。
可见性 `v` 的约定：

| v | 含义 | 坐标写法 |
| --- | --- | --- |
| 2 | 关键点落在画幅内（可见） | 原样输出归一化坐标 |
| 1 | 关键点有效但落在画幅外（出画/被裁） | 裁剪到画幅边界 |
| 0 | 无效（点在相机后方，无法投影） | `0.000000 0.000000` |

### 4.2 COCO-Keypoints（`keypoints.json`）

标准 COCO 结构（`images` / `annotations` / `categories`），并额外写入：

* `annotations[i].keypoints_world_m`：9 个关键点的**世界坐标(m)**，可与 `keypoints`
  构成 2D-3D 对应，直接用于 PnP / EPnP / 位姿回归；
* `annotations[i].target_pose`：目标机**位置(世界系) + 欧拉角(rad 与 deg)**；
* `keypoint_definition`：关键点顺序、机体坐标偏移、可见性约定；
* `camera`：内参矩阵、主点、焦距、相机世界位置/姿态、安装偏移。
* `categories[0].skeleton`：8 个顶点围成的长方体 12 条棱，便于可视化。

## 5. 9 个关键点的定义（唯一真值来源：`POINT_OFFSETS`）

关键点在**机体坐标系**下的偏移量（相对几何中心，单位 m，顺序 = 标签顺序）：

| 索引 | 名称 | x | y | z | 说明 |
| --- | --- | --- | --- | --- | --- |
| kp0 | belly_low | -0.035 | -0.035 | -0.150 | 机腹最低点（起落架） |
| kp1 | low_xp_yn | +0.245 | -0.245 | -0.035 | 下层顶点 |
| kp2 | low_xp_yp | +0.245 | +0.245 | -0.035 | 下层顶点 |
| kp3 | low_xn_yn | -0.245 | -0.245 | -0.035 | 下层顶点 |
| kp4 | low_xn_yp | -0.245 | +0.245 | -0.035 | 下层顶点 |
| kp5 | up_xp_yn | +0.130 | -0.130 | +0.170 | 上层顶点（桨盘以上） |
| kp6 | up_xp_yp | +0.130 | +0.130 | +0.170 | 上层顶点 |
| kp7 | up_xn_yn | -0.130 | -0.130 | +0.170 | 上层顶点 |
| kp8 | up_xn_yp | -0.130 | +0.130 | +0.170 | 上层顶点 |

包络盒：X 0.490 m × Y 0.490 m × Z 0.320 m。
“几何中心”位于 `sendUE4Pos` 指令位置下方 `copterCenterHeight = 0.15 m` 处。
**换机型时必须同步修改 `POINT_OFFSETS`**，否则标签与图像不符。

## 6. 训练建议

```bat
:: 1) 用生成的模板直接起训（Ultralytics）
yolo pose train data=dataset_pose.yaml model=yolo11n-pose.pt epochs=200 imgsz=640
```

* `kpt_shape: [9, 3]`，`names: {0: uav}`，已写入 `dataset_pose.yaml`。
* **务必关闭水平翻转增强**：本 9 点集合关于 y=0 平面**不对称**（kp0 的 x、y 偏移都是
  -0.035，其余点成对对称），镜像后没有任何关键点与之对应，`fliplr>0` 会直接产生错误标签。
  因此模板里**故意没有**配置 `flip_idx`，训练时请加 `fliplr=0.0`。
* 关键点监督建议：只对 `v==2` 的点算损失（`v==1/0` 的点坐标是裁剪/占位值，没有可靠语义）。
  若用 YOLO-Pose，它的 `kpt_loss` 默认会用到所有点，建议先把 `v<2` 的点过滤或另写 dataset 类。
* 位姿估计：`keypoints.json` 里的 2D-3D 对应足以做 PnP；若做端到端位姿回归，可把
  `target_pose` 作为监督，输入用图像（可选加检测框）。
* 数据量：每个采样周期只保存一张图，目标姿态随机范围与原始脚本一致
  （x∈[1.03,3.03]、y∈[-0.6,0.6]、z 变化 ±0.5，三轴姿态 ±20°/±20°/±30°）。
  若要覆盖更大的位姿范围，改主循环里的 `TargePos` / `TargeAng` 随机范围。
* `--min-keypoints N`：默认 4，即丢弃画幅内可见关键点不足 4 个的样本；想保留全部样本填 1。

## 7. 与原始脚本的差异（便于对照阅读）

1. 增加了关键点投影（带**深度保护**：点在相机后方时不会像原脚本那样出现除零/负深度伪坐标）；
2. 标签由“只有检测框”升级为“检测框 + 27 个关键点数值”；
3. 新增 `keypoints.json`、`dataset_pose.yaml`；
4. 新增配置项（命令行 / `Config.json`）：保存目录、时间戳子目录、最小可见点数、
   COCO 开关、headless、限量录制、类别名等；
5. 相机内参会按 `Config.json` 的 `DataWidth/DataHeight/CameraFOV/SensorPosXYZ` 自动覆盖，
   避免取图分辨率与标签假设不一致（默认配置下与原始脚本完全相同：640×480、focal=320）；
6. 几何代码做了单一真值来源重构（`getUav9Point()` 直接由 `POINT_OFFSETS` 生成），
   已用 `_verify_keypoints.py` 逐点校验与原脚本完全一致（误差 0）。

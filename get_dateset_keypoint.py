# -*- coding: utf-8 -*-
"""
get_dateset_keypoint.py  --  带“关键点(keypoint)录制”的数据集生成脚本
=====================================================================

本脚本由同目录下的 get_dateset.py 复制而来，完整保留原有流程：
    随机摆放目标无人机(vehicleID=100) -> 从 RflySim3D 取图 -> 反投影算出 9 个几何点
    -> 计算 YOLO 检测框标签 -> 保存 images/*.jpg + labels/*.txt

在此基础上新增了【关键点录制】功能，供后续训练关键点检测 / 位姿(SO3)估计神经网络：

1. 每保存一张图像，除检测框外，同时记录 9 个关键点的像素坐标与可见性；
2. 标签同时提供两种格式（可分别关闭）：

   (a) YOLO-Pose 格式：labels/{id}.txt，每行一个目标
       <cls> <cx> <cy> <w> <h> <k0x> <k0y> <k0v> ... <k8x> <k8y> <k8v>
       前 4 个框参数与 27 个关键点参数均归一化到 [0,1]（浮点），v 为可见性(int)：
           v = 2  关键点投影落在画幅内（可见，坐标未裁剪，原样输出）
           v = 1  关键点有效但落在画幅外（被裁掉/出画），坐标已裁剪到画幅边界
           v = 0  关键点无效（落在相机后方，无法投影），坐标写 0.000000 0.000000

   (b) COCO-Keypoints 格式：keypoints.json
       标准 COCO 字段 (images/annotations/categories) 之外，额外写入：
           annotations[i].keypoints_world_m  每个关键点的世界坐标(m)
           annotations[i].target_pose        目标机位置(世界系) + 欧拉角(rad/deg)
           keypoint_definition               关键点顺序与机体坐标偏移
           camera                            相机内参/外参
       便于做 2D-3D 对应、PnP、位姿回归等任务的监督。

3. 同时生成 dataset_pose.yaml（Ultralytics YOLO-Pose 训练模板）。

用法（与 get_dateset.py 一致，需先启动 RflySim3D 并等待 Grasslands 地图加载）：
    python get_dateset_keypoint.py                          # 保存到 Config.json 配置的目录
    python get_dateset_keypoint.py --save-dir D:/kp_data --no-timestamp
    python get_dateset_keypoint.py --min-keypoints 6        # 只保留可见关键点>=6 的样本
    python get_dateset_keypoint.py --no-coco                # 只写 YOLO-Pose 标签
    python get_dateset_keypoint.py --headless --max-samples 2000   # 无人值守录 2000 张

运行中把焦点放在 OpenCV 窗口上按 q 退出；退出时会自动把 keypoints.json 落盘并打印统计。

关键点顺序与机体坐标定义见下方 POINT_OFFSETS，与 get_dateset.py 的 getUav9Point()
完全一致（已由 _verify_keypoints.py 离线逐点校验）。
"""

import argparse
import datetime
import json
import math
import os
import random
import re
import sys
import time

import cv2
import numpy as np

# ----------------------------------------------------------------------
# RflySim 接口：放在 try 里，使得本文件在没有 RflySim 环境时也能被 import
# （_verify_keypoints.py 就是靠这一点做离线几何/格式自检的）
# ----------------------------------------------------------------------
try:
    import VisionCaptureApi
    import UE4CtrlAPI
    HAS_RFLYSIM = True
    _RFLYSIM_IMPORT_ERR = None
except Exception as _err:   # ImportError，或缺少其依赖
    HAS_RFLYSIM = False
    _RFLYSIM_IMPORT_ERR = _err


# ======================================================================
# 1. 目标无人机几何 —— 9 个关键点的定义（唯一的“真值”来源）
# ======================================================================
copterCenterHeight = 0.15       # “几何中心”位于 sendUE4Pos 指令位置下方多少米
UAVh = 0.185
UAVw = 0.185

# 9 个关键点在【机体坐标系】下的偏移量(m)，顺序 = 录制顺序 = 训练时的 keypoint 索引。
# 这组数值就是“目标无人机几何外形”的全部先验：换机型时必须同步修改这里。
# 注意：为了避免左右/前后命名歧义，这里只按坐标符号标注（x/y/z 的正负）。
POINT_OFFSETS = [
    [-0.035, -0.035, -0.135 - 0.015],   # kp0 机腹最低点(起落架)，z=-0.150
    [0.245, -0.245, -0.035],            # kp1 下层顶点 (x+, y-)
    [0.245, 0.245, -0.035],             # kp2 下层顶点 (x+, y+)
    [-0.245, -0.245, -0.035],           # kp3 下层顶点 (x-, y-)
    [-0.245, 0.245, -0.035],            # kp4 下层顶点 (x-, y+)
    [0.13, -0.13, 0.17],                # kp5 上层顶点 (x+, y-)，桨盘以上
    [0.13, 0.13, 0.17],                 # kp6 上层顶点 (x+, y+)
    [-0.13, -0.13, 0.17],               # kp7 上层顶点 (x-, y-)
    [-0.13, 0.13, 0.17],                # kp8 上层顶点 (x-, y+)
]
KEYPOINT_COUNT = len(POINT_OFFSETS)     # = 9

# 关键点名称：写入 COCO categories / dataset_pose.yaml 的注释，命名保持“无歧义”
KEYPOINT_NAMES = [
    'kp0_belly_low',
    'kp1_low_xp_yn', 'kp2_low_xp_yp', 'kp3_low_xn_yn', 'kp4_low_xn_yp',
    'kp5_up_xp_yn', 'kp6_up_xp_yp', 'kp7_up_xn_yn', 'kp8_up_xn_yp',
]

# 8 个顶点围成长方体的 12 条棱（kp1~kp8），用于可视化和 COCO skeleton
BOX_EDGES = [
    (1, 2), (2, 4), (4, 3), (3, 1),   # 下层四条棱（kp1~kp4，z=-0.035）
    (5, 6), (6, 8), (8, 7), (7, 5),   # 上层四条棱（kp5~kp8，z=+0.170）
    (1, 5), (2, 6), (3, 7), (4, 8),   # 连接上下的四条竖直棱
]
SKELETON = [list(e) for e in BOX_EDGES]

# 可见性编码（与 COCO keypoint 的 v 定义一致）
VIS_NOT_LABELED = 0     # 无效：点在相机后方 / 无法投影
VIS_OCCLUDED = 1        # 有效但出画（被裁掉）
VIS_VISIBLE = 2         # 在画幅内可见

# 关键点标志色（BGR）：kp0=青，kp1~4=蓝，kp5~8=红
KEYPOINT_COLORS = [
    (255, 255, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255),
]

# ======================================================================
# 2. 相机参数（默认值与 get_dateset.py 一致；若 Config.json 的 VisionSensors
#    段写了别的 DataWidth/DataHeight/CameraFOV，会在 main() 里按它覆盖）
# ======================================================================
pic_w = 640
pic_h = 480
focal = 320.0                   # 90 度 FOV / 640 宽 时，focal = (w/2)/tan(45) = 320
px = pic_w / 2.0                # 主点
py = pic_h / 2.0
cameraPosForUav = [0.03, 0, 0]  # 视觉传感器在机体坐标系下的安装位置(m)
MIN_DEPTH = 0.05                # 关键点深度(相机前方 X)小于该值视为无效

# 目标模型类型：与原脚本保持一致(3)。若你的目标机模型需要用 1（test_dataset.py 用的是 1），
# 改这里即可，不需要动其它代码。
TARGET_VEHICLE_TYPE = 3
TARGET_ID = 100

# ======================================================================
# 3. 本实验的开关与默认配置（都可用命令行 / Config.json 覆盖）
# ======================================================================
# ---- 可视化开关 ----
SHOW_KEYPOINTS = True        # 是否在预览窗口画出关键点/立体框
SHOW_KEYPOINT_DETAIL = True  # 是否画出 8 顶点围成的长方体棱边
SHOW_KEYPOINT_PERIOD = 1     # 每多少个采样周期刷新一次关键点预览窗口
PRINT_KEYPOINTS = False      # 是否在终端打印每个关键点的像素坐标（调试用）
KEYPOINT_LEGEND = 'kp0=belly(cyan) kp1-4=lower(blue) kp5-8=upper(red) hollow=out of frame'

# ---- 关键点标签开关 ----
ENABLE_KEYPOINTS = True      # 总开关：关闭后行为与原始 get_dateset.py 完全一致
MIN_VISIBLE_KEYPOINTS = 4    # 只保留“画幅内可见关键点”数量 >= 该值的样本；填 0/1 表示不过滤
BBOX_FROM_VISIBLE_ONLY = False   # True: 检测框只用画幅内的关键点算；False: 与原始脚本一致(所有有效点)
SAVE_COCO_JSON = True        # 是否额外写 COCO-Keypoints 格式的 keypoints.json
SAVE_DATASET_YAML = True     # 是否生成 Ultralytics YOLO-Pose 训练用的 dataset_pose.yaml
JSON_FLUSH_EVERY = 100       # 每保存多少张图就落盘一次 keypoints.json（防中途崩溃丢数据）
CLASS_ID = 0                 # YOLO 类别号
CLASS_NAME = 'uav'           # YOLO 类别名

# ---- 无界面/限量录制（批量、远程、自动化测试时用）----
HEADLESS = False             # True: 不创建/刷新 OpenCV 窗口（远程或无人值守录制）
MAX_SAMPLES = 0              # >0: 保存够这么多张就自动结束；0 = 一直录到按 q / Ctrl+C

# ---- 数据集保存位置配置 ----
# 优先级：命令行 --save-dir > 环境变量 SO3_DATASET_ROOT > Config.json 的 dataset 段 > 下面常量
DATASET_ROOT = ''            # '' = 使用脚本所在目录；也可填绝对/相对路径
USE_TIMESTAMP_SUBDIR = True  # True: <root>/20240101_120000/{images,labels,...}
IMAGE_SUBDIR = 'images'
LABEL_SUBDIR = 'labels'
CONFIG_FILE = 'Config.json'
COCO_JSON_NAME = 'keypoints.json'
DATASET_YAML_NAME = 'dataset_pose.yaml'


# ======================================================================
# 4. 基础几何 / 投影函数
# ======================================================================
def eul2rot(theta):
    '''
    欧拉角转旋转矩阵，与世界坐标系方向、欧拉角旋转顺序、旋转正方向的定义有关
    '''
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), math.sin(theta[0])],
                    [0, -math.sin(theta[0]), math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                    [0, 1, 0],
                    [math.sin(theta[1]), 0, math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]])

    R = np.dot(R_x, np.dot(R_y, R_z))
    return R


def getExternalMatrix(uavPos, cameraPosForUav, eurTheta):
    '''
    获取外参矩阵，转换为齐次坐标形式并输出
    '''
    rotMatrix = eul2rot(np.array(eurTheta))
    t = -np.dot(rotMatrix, (np.array(uavPos) +
                np.array(cameraPosForUav)).reshape(-1, 1))
    f = np.array([0, 0, 0, 1])
    rot_t = np.hstack((rotMatrix, t))
    ext_matrix = np.vstack((rot_t, f))
    return ext_matrix


def body2world(offset, uavPos, uavAng, centerHeight=None):
    '''
    机体坐标系偏移量 -> 世界坐标（与 get_dateset.py 中 getUav9Point 内部使用的是同一个变换）
    '''
    if centerHeight is None:
        centerHeight = copterCenterHeight
    center = np.array(uavPos, dtype=float) + [0, 0, -centerHeight]
    return center + np.transpose(
        np.dot(np.linalg.inv(eul2rot(uavAng)), np.transpose(np.array(offset, dtype=float))))


def getUav9Point(uavPos, copterCenterHeight_, uavAng, UAVh_=UAVh, UAVw_=UAVw,
                 eurTheta=[0, 0, 0]):
    '''
    通过输入 UAV 的坐标以及相关信息，获取九个空间点的坐标（顺序与 POINT_OFFSETS 严格一致）
    '''
    return [body2world(off, uavPos, uavAng, copterCenterHeight_) for off in POINT_OFFSETS]


def get_image_size(knownWidth, focalLength, distance):
    return (knownWidth * focalLength) / distance


def get_pic_situation(relative_pos, px_, py_, focalLength):
    '''
    原 get_dateset.py 的投影函数（按原样保留，便于对拍）。
    其数学本质就是小孔成像：u = px + f * Y / X，v = py + f * Z / X（X 为相机前方深度）。
    '''
    fx = (focalLength / relative_pos[0]) * \
        np.linalg.norm([relative_pos[0], relative_pos[2]])
    fy = (focalLength / relative_pos[0]) * \
        np.linalg.norm([relative_pos[0], relative_pos[1]])
    cx = get_image_size(relative_pos[1], fx, np.linalg.norm(
        [relative_pos[0], relative_pos[2]])) + px_
    cy = get_image_size(relative_pos[2], fy, np.linalg.norm(
        [relative_pos[0], relative_pos[1]])) + py_
    return [round(cx), round(cy)]


def relative_position(sit_a, sit_b):
    a = [sit_a[0] - sit_b[0], sit_a[1] - sit_b[1], sit_a[2] - sit_b[2]]
    return a


def get_distance(relative_pos):
    distance = np.linalg.norm(relative_pos)
    return distance


def xyxy2xywh(x1, x2):
    cent = [(x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2]
    wh = [x2[0] - x1[0], x2[1] - x1[1]]
    return cent, wh


def xywh2yolo(c, wh):
    y1 = round(c[0] / pic_w, 8)
    y2 = round(c[1] / pic_h, 8)
    y3 = round(wh[0] / pic_w, 8)
    y4 = round(wh[1] / pic_h, 8)
    return y1, y2, y3, y4


def world2camera(world_point, cam_pos, cam_ang):
    '''
    世界坐标 -> 相机坐标系（X 向前/光轴，Y、Z 与 get_pic_situation 的约定一致）
    '''
    rel = np.dot(eul2rot(cam_ang), np.transpose(
        relative_position(world_point, cam_pos)))
    return [float(rel[0]), float(rel[1]), float(rel[2])]


def project_point_pinhole(world_point, cam_pos, cam_ang,
                          img_w=None, img_h=None, focal_=None,
                          px_=None, py_=None, min_depth=MIN_DEPTH):
    '''
    【关键点录制核心】把世界坐标点投影成像素坐标，并给出可见性编码。

    返回 (u, v, depth, visibility)：
        u, v    浮点像素坐标（未裁剪；出画时会 <0 或 >=宽高）
        depth   相机前方深度 X(m)；无效时为 None
        visibility  见 VIS_* 三个常量
    '''
    if img_w is None:
        img_w = pic_w
    if img_h is None:
        img_h = pic_h
    if focal_ is None:
        focal_ = focal
    if px_ is None:
        px_ = px
    if py_ is None:
        py_ = py

    X, Y, Z = world2camera(world_point, cam_pos, cam_ang)
    if not np.isfinite([X, Y, Z]).all() or X <= min_depth:
        # 点在相机后方或深度过小：无法投影
        return 0.0, 0.0, None, VIS_NOT_LABELED

    u = px_ + focal_ * Y / X
    v = py_ + focal_ * Z / X
    inside = (0.0 <= u < float(img_w)) and (0.0 <= v < float(img_h))
    return float(u), float(v), float(X), (VIS_VISIBLE if inside else VIS_OCCLUDED)


def project_keypoints(world_points, cam_pos, cam_ang):
    '''
    批量投影，返回与 world_points 等长的 [(u, v, vis), ...] 以及深度列表
    '''
    kpts = []
    depths = []
    for wp in world_points:
        u, v, d, vis = project_point_pinhole(wp, cam_pos, cam_ang)
        kpts.append((u, v, vis))
        depths.append(d)
    return kpts, depths


def count_visible(kpts):
    '''
    统计落在画幅内的关键点个数
    '''
    return int(sum(1 for k in kpts if int(k[2]) == VIS_VISIBLE))


# ======================================================================
# 5. 标签构造 / 写盘
# ======================================================================
def _fmt(value):
    '''
    标签里的浮点数统一保留 6 位小数，避免过长且足够精度(<0.2 像素)
    '''
    return '{:.6f}'.format(float(value))


def _clamp(value, low, high):
    return low if value < low else (high if value > high else value)


def build_yolo_pose_line(class_id, bbox_xyxy, kpts):
    '''
    构造一行 YOLO-Pose 标签：
        <cls> <cx> <cy> <w> <h> <k0x> <k0y> <k0v> ... <k8x> <k8y> <k8v>
    全部归一化到 [0,1]。出画的关键点坐标裁剪到画幅边界并置 v=1；无效点置 0 0 0。
    '''
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    cx = (x1 + x2) / 2.0 / pic_w
    cy = (y1 + y2) / 2.0 / pic_h
    bw = max(0.0, x2 - x1) / pic_w
    bh = max(0.0, y2 - y1) / pic_h

    parts = [str(int(class_id)), _fmt(cx), _fmt(cy), _fmt(bw), _fmt(bh)]
    for u, v, vis in kpts:
        vis = int(vis)
        if vis == VIS_NOT_LABELED:
            parts += ['0.000000', '0.000000', '0']
        else:
            nu = _clamp(float(u), 0.0, pic_w) / pic_w
            nv = _clamp(float(v), 0.0, pic_h) / pic_h
            parts += [_fmt(nu), _fmt(nv), str(vis)]
    return ' '.join(parts) + '\n'


def build_coco_annotation(ann_id, image_id, bbox_xyxy, kpts, world_points=None,
                          target_pos=None, target_ang=None):
    '''
    构造一条 COCO keypoints 标注（含额外写入的 3D 信息与目标位姿）
    '''
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    flat = []
    for u, v, vis in kpts:
        flat += [round(float(u), 3), round(float(v), 3), int(vis)]

    ann = {
        'id': int(ann_id),
        'image_id': int(image_id),
        'category_id': 1,
        'bbox': [round(x1, 3), round(y1, 3), round(max(0.0, x2 - x1), 3),
                 round(max(0.0, y2 - y1), 3)],
        'area': round(max(0.0, x2 - x1) * max(0.0, y2 - y1), 3),
        'iscrowd': 0,
        'num_keypoints': count_visible(kpts),
        'keypoints': flat,
    }
    if world_points is not None:
        ann['keypoints_world_m'] = [[round(float(p[0]), 4), round(float(p[1]), 4),
                                     round(float(p[2]), 4)] for p in world_points]
    if target_pos is not None:
        ann['target_pose'] = {
            'position_world_m': [round(float(v), 4) for v in target_pos],
            'euler_rad': [round(float(v), 6) for v in target_ang],
            'euler_deg': [round(math.degrees(float(v)), 3) for v in target_ang],
        }
    return ann


def build_coco_doc(images, annotations, camera_info=None):
    '''
    组装完整 COCO-Keypoints 文档
    '''
    doc = {
        'info': {
            'description': 'RflySim SO3 keypoint dataset (get_dateset_keypoint.py)',
            'date_created': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task': 'keypoints',
        },
        'licenses': [],
        'keypoint_definition': {
            'count': KEYPOINT_COUNT,
            'names': KEYPOINT_NAMES,
            'offsets_body_m': [list(o) for o in POINT_OFFSETS],
            'body_frame_note': 'offset = R(attitude) @ (point_world - (target_pos + [0,0,-copterCenterHeight]))',
            'copter_center_height_m': copterCenterHeight,
            'visibility': {'0': 'invalid/behind camera', '1': 'valid but outside image',
                           '2': 'inside image'},
        },
        'images': images,
        'annotations': annotations,
        'categories': [{
            'id': 1,
            'name': CLASS_NAME,
            'supercategory': 'uav',
            'keypoints': KEYPOINT_NAMES,
            'skeleton': SKELETON,
        }],
    }
    if camera_info is not None:
        doc['camera'] = camera_info
    return doc


def build_camera_info(cam_pos, cam_ang):
    '''
    记录观测相机的内参与外参，方便后续做 PnP / 重投影验证
    '''
    return {
        'image_width': int(pic_w),
        'image_height': int(pic_h),
        'focal_px': float(focal),
        'principal_point': [float(px), float(py)],
        'intrinsics_3x3': [[float(focal), 0.0, float(px)],
                           [0.0, float(focal), float(py)],
                           [0.0, 0.0, 1.0]],
        'position_world_m': [round(float(v), 4) for v in cam_pos],
        'attitude_euler_rad': [round(float(v), 6) for v in cam_ang],
        'sensor_offset_body_m': [float(v) for v in cameraPosForUav],
        'projection_note': 'camera frame X=forward(optical axis), u=px+f*Y/X, v=py+f*Z/X',
    }


def write_json_atomic(path, doc):
    '''
    先写临时文件再替换，避免中途崩溃留下半个 JSON
    '''
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False)
    if os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)


def build_dataset_yaml(path_dir, image_subdir, class_id, class_name):
    '''
    生成 Ultralytics YOLO-Pose 训练模板。
    注意：这里故意不写 flip_idx —— 该 9 点集合（kp0 的 x、y 偏移都是 -0.035）
    关于 y=0 平面并不对称，水平翻转增强会破坏关键点语义，请在训练时关闭 flipud/fliplr。
    '''
    path_value = os.path.abspath(path_dir).replace('\\', '/')
    lines = [
        '# 由 get_dateset_keypoint.py 自动生成：Ultralytics YOLO-Pose 数据集配置',
        '# 用法: yolo pose train data=%s model=yolo11n-pose.pt' % DATASET_YAML_NAME,
        '',
        'path: %s' % path_value,
        'train: %s' % image_subdir,
        'val: %s' % image_subdir,
        '',
        '# 每个目标 9 个关键点，每个关键点 3 个数 (x, y, visibility)',
        'kpt_shape: [%d, 3]' % KEYPOINT_COUNT,
        '',
        'names:',
        '  %d: %s' % (int(class_id), class_name),
        '',
        '# 关键点顺序（与 labels/*.txt 中的顺序一致）：',
    ]
    for i, name in enumerate(KEYPOINT_NAMES):
        lines.append('#   %d: %s   body_offset(m) = %s'
                     % (i, name, POINT_OFFSETS[i]))
    lines += [
        '',
        '# 重要：本关键点集合不是左右对称的（kp0 的 y 偏移为 -0.035），',
        '#       因此训练时请关闭水平翻转增强（fliplr=0.0），也不要在 yaml 里配置 flip_idx。',
        '# 未使用到的关键点（出画/相机后方）的可见性: 2=可见, 1=出画, 0=无效',
        '',
    ]
    return '\n'.join(lines)


# ======================================================================
# 6. 可视化
# ======================================================================
def draw_keypoints(img, kpts, bbox_xyxy=None, detail=True):
    '''
    在一张图上画出 9 个关键点：
        kp0=青色(机腹最低点)，kp1~4=蓝色(下层顶点)，kp5~8=红色(上层顶点)
        实心圆 = 在画幅内(v=2)；空心圆 + '*' = 出画(v=1)；v=0 不画点
    可选画出 8 个顶点围成的长方体棱边（只画两端都有效的棱）。
    '''
    out = img.copy()
    pts = [(int(round(k[0])), int(round(k[1]))) for k in kpts]

    if detail:
        for i, j in BOX_EDGES:
            if int(kpts[i][2]) >= VIS_OCCLUDED and int(kpts[j][2]) >= VIS_OCCLUDED:
                cv2.line(out, pts[i], pts[j], (0, 255, 0), 1, cv2.LINE_AA)

    if bbox_xyxy is not None:
        cv2.rectangle(out, (int(round(bbox_xyxy[0])), int(round(bbox_xyxy[1]))),
                      (int(round(bbox_xyxy[2])), int(round(bbox_xyxy[3]))),
                      (0, 0, 255), 1)

    for i, (u, v, vis) in enumerate(kpts):
        vis = int(vis)
        if vis == VIS_NOT_LABELED:
            continue
        color = KEYPOINT_COLORS[i] if i < len(KEYPOINT_COLORS) else (255, 255, 255)
        p = (int(round(u)), int(round(v)))
        tag = str(i) if vis == VIS_VISIBLE else str(i) + '*'
        if vis == VIS_VISIBLE:
            cv2.circle(out, p, 4, color, -1, cv2.LINE_AA)          # 实心 = 可见
        else:
            cv2.circle(out, p, 4, color, 1, cv2.LINE_AA)           # 空心 = 出画
        cv2.circle(out, p, 6, (255, 255, 255), 1, cv2.LINE_AA)     # 白圈，便于辨认
        org = (p[0] + 8, p[1] - 8)
        cv2.putText(out, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(out, KEYPOINT_LEGEND, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, KEYPOINT_LEGEND, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_console_safe():
    '''
    本脚本的提示信息含中文。若控制台代码页不是 GBK/UTF-8（例如英文系统的 cp437），
    print 会抛 UnicodeEncodeError。这里把 errors 改成 replace：能显示就显示，不能显示就
    用 '?' 代替，绝不因为一行日志让录制中断。
    '''
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(errors='replace')
            except Exception:
                pass


def show_window(name, image, resize_to=None):
    '''
    统一的预览显示入口：HEADLESS=True 时什么都不做（便于远程/无人值守录制）
    '''
    if HEADLESS:
        return
    if resize_to is not None:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_NEAREST)
    cv2.imshow(name, image)


def poll_quit():
    '''
    统一的按键检查：HEADLESS=True 时永远返回 False（只能用 --max-samples 或 Ctrl+C 结束）
    '''
    if HEADLESS:
        return False
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ======================================================================
# 7. 配置文件 / 命令行解析
# ======================================================================
def strip_json_comments(text):
    '''
    去掉 Config.json 里的 // 与 # 注释（与 VisionCaptureApi.jsonLoad 的做法保持一致）
    '''
    lines = []
    for row in text.splitlines():
        if row.strip().startswith('//') or row.strip().startswith('#'):
            continue
        row = re.sub('//.*', '', row)
        row = re.sub('#.*', '', row)
        lines.append(row)
    return '\n'.join(lines)


def load_config_json(config_path):
    '''
    读取并解析 Config.json（失败返回空 dict，绝不影响主流程）
    '''
    if not config_path or not os.path.isfile(config_path):
        print('[配置] 未找到 {}，将使用脚本内默认参数'.format(config_path))
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            js_data = json.loads(strip_json_comments(f.read()))
        return js_data if isinstance(js_data, dict) else {}
    except Exception as err:
        print('[警告] 解析 {} 失败({})，将使用脚本内默认参数'.format(config_path, err))
        return {}


def _get_ci(container, *names):
    '''
    大小写不敏感地从 dict 里取第一个存在的键
    '''
    if not isinstance(container, dict):
        return None
    lower = {re.sub(r'[_\-\s]', '', str(k)).lower(): v for k, v in container.items()}
    for name in names:
        key = re.sub(r'[_\-\s]', '', str(name)).lower()
        if key in lower:
            return lower[key]
    return None


def _to_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_dataset_cfg(js_data):
    '''
    取出 Config.json 的 dataset 段（兼容对象与对象数组两种写法）
    '''
    ds = js_data.get('dataset', {}) if isinstance(js_data, dict) else {}
    if isinstance(ds, list):
        merged = {}
        for item in ds:
            if isinstance(item, dict):
                merged.update(item)
        ds = merged
    return ds if isinstance(ds, dict) else {}


def cfg_get_str(cfg, *names):
    for name in names:
        value = _get_ci(cfg, name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def cfg_get_bool(cfg, *names):
    for name in names:
        value = _get_ci(cfg, name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ('true', '1', 'yes', 'y', 'on'):
                return True
            if text in ('false', '0', 'no', 'n', 'off'):
                return False
    return None


def cfg_get_int(cfg, *names):
    for name in names:
        value = _get_ci(cfg, name)
        num = _to_int(value)
        if num is not None:
            return num
    return None


def parse_args():
    '''
    解析命令行参数；用 parse_known_args 兼容 RflySim 等外部调用时附带的额外参数
    '''
    parser = argparse.ArgumentParser(
        description='生成带关键点(keypoint)标签的 SO3 视觉数据集')
    parser.add_argument('--save-dir', '--save_dir', dest='save_dir', default=None,
                        help='数据集保存根目录，优先于环境变量 SO3_DATASET_ROOT、Config.json 和脚本内常量')
    parser.add_argument('--config', dest='config', default=None,
                        help='配置文件路径（默认脚本目录下的 Config.json）')
    parser.add_argument('--timestamp', dest='timestamp', action='store_true',
                        default=None, help='在根目录下创建时间戳子目录（默认行为）')
    parser.add_argument('--no-timestamp', '--no_timestamp', dest='timestamp',
                        action='store_false', default=None,
                        help='不创建时间戳子目录，直接写入 <根目录>/images 与 <根目录>/labels')
    parser.add_argument('--min-keypoints', '--min_keypoints', dest='min_keypoints',
                        type=int, default=None,
                        help='只保留画幅内可见关键点数量 >= 该值的样本（默认 %d）'
                             % MIN_VISIBLE_KEYPOINTS)
    parser.add_argument('--no-coco', '--no_coco', dest='save_coco', action='store_false',
                        default=None, help='不写 COCO-Keypoints 格式的 keypoints.json')
    parser.add_argument('--no-keypoints', '--no_keypoints', dest='keypoints',
                        action='store_false', default=None,
                        help='关闭关键点功能（标签退化为原始 YOLO 检测框格式）')
    parser.add_argument('--print-keypoints', '--print_keypoints', dest='print_kpts',
                        action='store_true', default=None,
                        help='在终端打印每个关键点的像素坐标（调试用）')
    parser.add_argument('--headless', dest='headless', action='store_true', default=None,
                        help='不显示 OpenCV 预览窗口（远程/无人值守录制用）')
    parser.add_argument('--max-samples', '--max_samples', dest='max_samples', type=int,
                        default=None, help='保存够 N 张样本后自动结束（0=不限，默认 0）')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('[提示] 已忽略无法识别的命令行参数: {}'.format(unknown))
    return args


def resolve_save_root(cli_save_dir, env_save_dir, cfg_save_dir, script_dir):
    '''
    按 命令行 > 环境变量 > Config.json > 脚本内常量 > 脚本目录 的优先级确定数据集根目录
    '''
    candidates = [
        ('命令行参数 --save-dir', cli_save_dir),
        ('环境变量 SO3_DATASET_ROOT', env_save_dir),
        ('Config.json 的 dataset.SaveDir', cfg_save_dir),
        ('脚本内 DATASET_ROOT', DATASET_ROOT),
        ('脚本所在目录（默认）', script_dir),
    ]
    source, root = candidates[-1]
    for src, value in candidates:
        if value is not None and value != '':
            source, root = src, value
            break
    root = os.path.expanduser(os.path.expandvars(str(root)))
    if not os.path.isabs(root):
        root = os.path.join(script_dir, root)
    print('[配置] 数据集保存根目录来源: {}'.format(source))
    return os.path.normpath(root)


def apply_camera_config(js_data):
    '''
    若 Config.json 的 VisionSensors 段写了图像尺寸/FOV/安装位置，则用它覆盖脚本内的默认内参，
    保证关键点标签与真实取图分辨率一致。
    '''
    global pic_w, pic_h, focal, px, py, cameraPosForUav
    sensors = _get_ci(js_data, 'VisionSensors', 'visionSensors')
    sensor = None
    if isinstance(sensors, list) and sensors:
        sensor = sensors[0]
    elif isinstance(sensors, dict):
        sensor = sensors
    if isinstance(sensor, dict):
        w = _to_int(_get_ci(sensor, 'DataWidth'))
        h = _to_int(_get_ci(sensor, 'DataHeight'))
        fov = _to_float(_get_ci(sensor, 'CameraFOV'))
        pos = _get_ci(sensor, 'SensorPosXYZ')
        if w and w > 0:
            pic_w = int(w)
        if h and h > 0:
            pic_h = int(h)
        if fov and 0.0 < fov < 180.0:
            # 90 度 FOV + 640 宽 => focal = 320，与脚本默认值一致
            focal = (pic_w / 2.0) / math.tan(math.radians(fov) / 2.0)
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            cameraPosForUav = [float(v) for v in pos]
    px = pic_w / 2.0
    py = pic_h / 2.0
    print('[配置] 关键点投影内参: %dx%d focal=%.2f px=%.1f py=%.1f sensor_offset=%s'
          % (pic_w, pic_h, focal, px, py, cameraPosForUav))


def print_keypoint_definition():
    print('=== 目标机 %d 个关键点（相对几何中心的机体坐标，单位 cm）===' % KEYPOINT_COUNT)
    print('    说明：copterCenterHeight = %.2f m 表示“几何中心”位于 sendUE4Pos 指令位置下方多远'
          % copterCenterHeight)
    for k, off in enumerate(POINT_OFFSETS):
        print('    kp%-2d %-14s x=%+7.2f  y=%+7.2f  z=%+7.2f'
              % (k, KEYPOINT_NAMES[k], off[0] * 100, off[1] * 100, off[2] * 100))
    print('    包络盒尺寸：X %.3f m  Y %.3f m  Z %.3f m'
          % (max(o[0] for o in POINT_OFFSETS) - min(o[0] for o in POINT_OFFSETS),
             max(o[1] for o in POINT_OFFSETS) - min(o[1] for o in POINT_OFFSETS),
             max(o[2] for o in POINT_OFFSETS) - min(o[2] for o in POINT_OFFSETS)))


# ======================================================================
# 8. 主流程
# ======================================================================
def main():
    global MIN_VISIBLE_KEYPOINTS, SAVE_COCO_JSON, ENABLE_KEYPOINTS, \
        PRINT_KEYPOINTS, CLASS_ID, CLASS_NAME, HEADLESS, MAX_SAMPLES

    make_console_safe()

    script_dir = sys.path[0] or os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(script_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))

    args = parse_args()

    # ---- 8.1 读取 Config.json（相机内参 + dataset 保存配置）----
    config_path = args.config or CONFIG_FILE
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    config_path = os.path.normpath(config_path)
    js_data = load_config_json(config_path)
    apply_camera_config(js_data)
    dataset_cfg = get_dataset_cfg(js_data)

    # ---- 8.2 命令行 > Config.json > 脚本内默认值 ----
    if args.keypoints is not None:
        ENABLE_KEYPOINTS = args.keypoints
    if args.save_coco is not None:
        SAVE_COCO_JSON = args.save_coco
    if args.print_kpts is not None:
        PRINT_KEYPOINTS = args.print_kpts
    if args.min_keypoints is not None:
        MIN_VISIBLE_KEYPOINTS = args.min_keypoints
    else:
        cfg_min = cfg_get_int(dataset_cfg, 'MinKeypoints', 'MinVisibleKeypoints',
                              'MinKeypointNum')
        if cfg_min is not None:
            MIN_VISIBLE_KEYPOINTS = cfg_min
    MIN_VISIBLE_KEYPOINTS = max(0, int(MIN_VISIBLE_KEYPOINTS))

    cfg_coco = cfg_get_bool(dataset_cfg, 'SaveCocoJson', 'SaveCoco', 'CocoJson')
    if args.save_coco is None and cfg_coco is not None:
        SAVE_COCO_JSON = cfg_coco
    cfg_cls_name = cfg_get_str(dataset_cfg, 'ClassName', 'CategoryName')
    if cfg_cls_name:
        CLASS_NAME = cfg_cls_name
    cfg_cls_id = cfg_get_int(dataset_cfg, 'ClassId', 'CategoryId')
    if cfg_cls_id is not None:
        CLASS_ID = cfg_cls_id
    if args.headless is not None:
        HEADLESS = args.headless
    if args.max_samples is not None:
        MAX_SAMPLES = max(0, int(args.max_samples))

    # ---- 8.3 目录 ----
    save_root = resolve_save_root(args.save_dir, os.environ.get('SO3_DATASET_ROOT'),
                                  cfg_get_str(dataset_cfg, 'SaveDir', 'SavePath',
                                              'DatasetRoot'), script_dir)
    cfg_ts = cfg_get_bool(dataset_cfg, 'UseTimestampSubdir', 'UseTimestamp', 'Timestamp')
    use_timestamp = args.timestamp if args.timestamp is not None else (
        cfg_ts if cfg_ts is not None else USE_TIMESTAMP_SUBDIR)
    image_subdir = cfg_get_str(dataset_cfg, 'ImageSubdir', 'ImgSubdir') or IMAGE_SUBDIR
    label_subdir = cfg_get_str(dataset_cfg, 'LabelSubdir', 'LabSubdir') or LABEL_SUBDIR

    path_dir = os.path.join(save_root, datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) \
        if use_timestamp else save_root
    os.makedirs(path_dir, exist_ok=True)
    path_img = os.path.join(path_dir, image_subdir)
    labels = os.path.join(path_dir, label_subdir)
    os.makedirs(path_img, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    coco_path = os.path.join(path_dir, COCO_JSON_NAME)

    print('config.json : {}'.format(config_path))
    print('save_root   : {}'.format(save_root))
    print('path_dir    : {}'.format(path_dir))
    print('images      : {}'.format(path_img))
    print('labels      : {}'.format(labels))
    print('use_timestamp: {}，min_visible_keypoints: {}，save_coco_json: {}，headless: {}，max_samples: {}'
          .format(use_timestamp, MIN_VISIBLE_KEYPOINTS, SAVE_COCO_JSON, HEADLESS, MAX_SAMPLES))

    print_keypoint_definition()

    if SAVE_DATASET_YAML:
        yaml_path = os.path.join(path_dir, DATASET_YAML_NAME)
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(build_dataset_yaml(path_dir, image_subdir, CLASS_ID, CLASS_NAME))
            print('[关键点] 已生成训练配置: {}'.format(yaml_path))
        except Exception as err:
            print('[警告] 写 {} 失败: {}'.format(yaml_path, err))

    if not ENABLE_KEYPOINTS:
        print('[关键点] ENABLE_KEYPOINTS=False：标签退化为原始 YOLO 检测框格式')

    # ---- 8.4 连接 RflySim3D ----
    if not HAS_RFLYSIM:
        print('[错误] 未能导入 VisionCaptureApi / UE4CtrlAPI（{}）。'.format(_RFLYSIM_IMPORT_ERR))
        print('       请在已配置好 RflySim 的 Python 环境中运行本脚本（见 Python38Run.bat）。')
        return 1

    ue = UE4CtrlAPI.UE4CtrlAPI()
    vis = VisionCaptureApi.VisionCaptureApi()

    ue.sendUE4Cmd('RflyChangeMapbyName Grasslands')
    ue.waitForMapLoaded('Grasslands')
    time.sleep(5)

    # 设置UE4窗口分辨率，注意本窗口仅限于显示，取图分辨率在json中配置，本窗口设置越小，资源需求越少。
    ue.sendUE4Cmd('r.setres 720x405w', 0)
    ue.sendUE4Cmd('t.MaxFPS 30', 0)  # 设置UE4最大刷新频率，同时也是取图频率
    time.sleep(2)

    # 创建观测用载具(充当相机)，初始位置在地面上方 4m
    PosInit = [0, 0, -8.086 - 4]
    angCopterE = [0, 0, 0]
    ue.sendUE4Pos(1, 0, 0, PosInit, angCopterE)
    time.sleep(2)

    vis.jsonLoad()  # 加载Config.json中的传感器配置文件
    isSuss = vis.sendReqToUE4()  # 向RflySim3D发送取图请求，并验证
    if not isSuss:  # 如果请求取图失败，则退出
        print('[错误] 取图请求失败，请检查 RflySim3D 是否已启动。')
        return 1
    vis.startImgCap(True)  # 开启取图，并启用共享内存图像转发
    time.sleep(1)

    # 摆放目标无人机(vehicleID=100)
    InitTargePos = [1.03, 0, -8.086 - 4]
    InitTargeAng = [0, 0, math.pi / 2]
    ue.sendUE4Pos(TARGET_ID, TARGET_VEHICLE_TYPE, 0, InitTargePos, InitTargeAng)
    time.sleep(1)

    if SHOW_KEYPOINTS and not HEADLESS:
        cv2.namedWindow('Keypoints', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Keypoints', 960, 720)  # 放大显示，取图分辨率仍由Config.json决定

    # 观测相机的世界位置（固定不动）
    camere_pos = PosInit + np.transpose(np.dot(
        np.linalg.inv(eul2rot(angCopterE)), np.transpose(cameraPosForUav)))

    images = []
    annotations = []
    stat = {'cycles': 0, 'saved': 0, 'skip_out_of_frame': 0, 'skip_low_vis': 0,
            'skip_no_data': 0}
    startTime = time.time()
    lastTime = time.time()
    timeInterval = 0.1
    cnt = 0
    num = 0
    camera_info = build_camera_info(camere_pos, angCopterE)

    def flush_coco():
        if not SAVE_COCO_JSON:
            return
        try:
            write_json_atomic(coco_path, build_coco_doc(images, annotations, camera_info))
        except Exception as err:
            print('[警告] 写 {} 失败: {}'.format(coco_path, err))

    print('[关键点] 开始录制，按 q（焦点在 OpenCV 窗口）退出并落盘。'
          + ('（headless 模式，保存 %d 张后自动结束）' % MAX_SAMPLES if MAX_SAMPLES > 0 else ''))
    try:
        while True:
            lastTime = lastTime + timeInterval
            sleepTime = lastTime - time.time()
            if sleepTime > 0:
                time.sleep(sleepTime)
            else:
                lastTime = time.time()

            num = num + 1
            # 每 0.5s 随机换一次目标位姿
            if num % 5 == 0:
                stat['cycles'] += 1
                TargePos = [InitTargePos[0] + random.randint(0, 200) / 100.0,
                            InitTargePos[1] + random.randint(-60, 60) / 100.0,
                            InitTargePos[2] + random.randint(-100, 100) / 200.0]
                TargeAng = [InitTargeAng[0] + random.randint(-50, 50) / 50.0 * 20 / 180.0 * math.pi,
                            InitTargeAng[1] + random.randint(-50, 50) / 50.0 * 20 / 180.0 * math.pi,
                            InitTargeAng[2] + random.randint(-50, 50) / 50.0 * 30 / 180.0 * math.pi]

                ue.sendUE4Pos(TARGET_ID, TARGET_VEHICLE_TYPE, 0, TargePos, TargeAng, -1)
                world_points = getUav9Point(TargePos, copterCenterHeight, TargeAng)
                time.sleep(0.2)

                if vis.hasData[0]:
                    show_window('pic1', vis.Img[0])
                    poll_quit()

                # ---- 关键点投影：世界坐标 -> 像素坐标 + 可见性 ----
                kpts, depths = project_keypoints(world_points, camere_pos, angCopterE)
                visible_count = count_visible(kpts)

                if PRINT_KEYPOINTS:
                    print('--- 关键点像素坐标 (u, v, vis)，画幅 %dx%d ---' % (pic_w, pic_h))
                    for k in range(KEYPOINT_COUNT):
                        print('  %-14s -> (%8.2f, %8.2f)  vis=%d  depth=%s'
                              % (KEYPOINT_NAMES[k], kpts[k][0], kpts[k][1], kpts[k][2],
                                 ('%.3f' % depths[k]) if depths[k] is not None else 'None'))

                # ---- 检测框：与原始脚本一致地用“所有有效点”的外接矩形 ----
                if BBOX_FROM_VISIBLE_ONLY:
                    used = [(u, v) for (u, v, vis) in kpts if int(vis) == VIS_VISIBLE]
                else:
                    used = [(u, v) for (u, v, vis) in kpts if int(vis) >= VIS_OCCLUDED]

                if not used:
                    stat['skip_out_of_frame'] += 1
                    cnt += 1
                    if poll_quit():
                        break
                    continue

                x1y1 = [min(p[0] for p in used), min(p[1] for p in used)]
                x2y2 = [max(p[0] for p in used), max(p[1] for p in used)]

                target_in_frame = (0 < (x1y1[0] + x2y2[0]) / 2 < pic_w) and \
                                  (0 < (x1y1[1] + x2y2[1]) / 2 < pic_h)

                # 裁剪到画幅内（与原脚本一致）
                if target_in_frame:
                    x1y1[0] = _clamp(x1y1[0], 0, pic_w)
                    x1y1[1] = _clamp(x1y1[1], 0, pic_h)
                    x2y2[0] = _clamp(x2y2[0], 0, pic_w)
                    x2y2[1] = _clamp(x2y2[1], 0, pic_h)

                keep = target_in_frame and (visible_count >= MIN_VISIBLE_KEYPOINTS)

                if vis.hasData[0]:
                    img1 = vis.Img[0]

                    if keep:
                        # ---- 保存原始图像（干净、不带任何绘制）----
                        img_name = '{}.jpg'.format(cnt)
                        cv2.imwrite(os.path.join(path_img, img_name), img1)

                        # ---- 保存 YOLO-Pose 关键点标签 ----
                        bbox = [x1y1[0], x1y1[1], x2y2[0], x2y2[1]]
                        if ENABLE_KEYPOINTS:
                            line = build_yolo_pose_line(CLASS_ID, bbox, kpts)
                        else:
                            cent, wh = xyxy2xywh(x1y1, x2y2)
                            y1, y2, y3, y4 = xywh2yolo(cent, wh)
                            line = '0 {} {} {} {}\n'.format(y1, y2, y3, y4)
                        with open(os.path.join(labels, '{}.txt'.format(cnt)), 'w') as f:
                            f.write(line)

                        # ---- 记录 COCO 关键点标注（含 3D 坐标与目标位姿）----
                        if SAVE_COCO_JSON and ENABLE_KEYPOINTS:
                            images.append({'id': int(cnt), 'file_name': img_name,
                                           'width': int(pic_w), 'height': int(pic_h)})
                            annotations.append(build_coco_annotation(
                                len(annotations) + 1, cnt, bbox, kpts,
                                world_points=world_points,
                                target_pos=TargePos, target_ang=TargeAng))

                        stat['saved'] += 1
                        if SAVE_COCO_JSON and ENABLE_KEYPOINTS and \
                                stat['saved'] % max(1, JSON_FLUSH_EVERY) == 0:
                            flush_coco()
                            print('[关键点] 已保存 %d 张（keypoints.json 已落盘）'
                                  % stat['saved'])
                        if MAX_SAMPLES > 0 and stat['saved'] >= MAX_SAMPLES:
                            print('[关键点] 已达到 --max-samples %d，自动结束录制。'
                                  % MAX_SAMPLES)
                            break
                    else:
                        if not target_in_frame:
                            stat['skip_out_of_frame'] += 1
                        else:
                            stat['skip_low_vis'] += 1

                    # ---- 预览：画出关键点/立体框/检测框 ----
                    img_show = img1.copy()
                    if ENABLE_KEYPOINTS:
                        if SHOW_KEYPOINTS and (stat['cycles'] % max(1, SHOW_KEYPOINT_PERIOD) == 0):
                            img_show = draw_keypoints(
                                img_show, kpts, bbox_xyxy=[x1y1[0], x1y1[1], x2y2[0], x2y2[1]],
                                detail=SHOW_KEYPOINT_DETAIL)
                            show_window('Keypoints', img_show, resize_to=(960, 720))
                        else:
                            cv2.rectangle(img_show, (int(x1y1[0]), int(x1y1[1])),
                                          (int(x2y2[0]), int(x2y2[1])), (0, 0, 255))
                    else:
                        cv2.rectangle(img_show, (int(x1y1[0]), int(x1y1[1])),
                                      (int(x2y2[0]), int(x2y2[1])), (0, 0, 255))
                    show_window('pic1', img_show)
                else:
                    stat['skip_no_data'] += 1

                # add image storing algorithm for MATLAB offline calibration
                # or add online calibration algorithm here
                cnt += 1

            if poll_quit():
                break
    except KeyboardInterrupt:
        print('\n[关键点] 收到 Ctrl+C，正在落盘 ...')
    finally:
        flush_coco()
        elapsed = time.time() - startTime
        print('=' * 60)
        print('[汇总] 用时 %.1f s，采样周期 %d，保存样本 %d'
              % (elapsed, stat['cycles'], stat['saved']))
        print('       跳过：目标出画 %d，可见点不足 %d，无图像数据 %d'
              % (stat['skip_out_of_frame'], stat['skip_low_vis'], stat['skip_no_data']))
        if SAVE_COCO_JSON and ENABLE_KEYPOINTS:
            print('       COCO 关键点文件: {}'.format(coco_path))
        print('       图像目录: {}'.format(path_img))
        print('       标签目录: {}'.format(labels))
        print('=' * 60)
        if not HEADLESS:
            cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())

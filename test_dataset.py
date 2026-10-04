# import required libraries
import random
import os
import re
import json
import argparse
from turtle import shape
import cv2
import time
import math
import sys
import datetime
import numpy as np

import VisionCaptureApi
import UE4CtrlAPI

ue = UE4CtrlAPI.UE4CtrlAPI()
vis = VisionCaptureApi.VisionCaptureApi()

ue.sendUE4Cmd('RflyChangeMapbyName Grasslands')
ue.waitForMapLoaded('Grasslands')
time.sleep(5)

# Send command to UE4 Window 1 to change resolution
# 设置UE4窗口分辨率，注意本窗口仅限于显示，取图分辨率在json中配置，本窗口设置越小，资源需求越少。
ue.sendUE4Cmd('r.setres 720x405w', 0)
ue.sendUE4Cmd('t.MaxFPS 30', 0)  # 设置UE4最大刷新频率，同时也是取图频率

time.sleep(2)

# send position command to create a vehicle (to simulate a camera)
# Initalize it to the 1.5m above the ground surface (the ground of Grassland map is -8.086)
PosInit = [0, 0, -8.086-4]
angCopterE = [0, 0, 0]
ue.sendUE4Pos(1, 0, 0, PosInit, angCopterE)
time.sleep(2)

# VisionCaptureApi 中的配置函数
vis.jsonLoad()  # 加载Config.json中的传感器配置文件

isSuss = vis.sendReqToUE4()  # 向RflySim3D发送取图请求，并验证
if not isSuss:  # 如果请求取图失败，则退出
    sys.exit(0)
vis.startImgCap(True)  # 开启取图，并启用共享内存图像转发，转发到填写的目录

time.sleep(1)

# send position command to create a checkerboard with vehicleID=100
# the checkerboard is located to 1m before the vehicle, with yaw angle 90 degree (face the vhicle)
InitTargePos = [1.03, 0, -8.086-4]
InitTargeAng = [0, 0, math.pi/2]
ue.sendUE4Pos(100, 1, 0, InitTargePos, InitTargeAng)
# 相机标定后的内参矩阵
intrMatrix = np.matrix([[320, 0, 0],
                        [0, 320, 0],
                        [320, 240, 1]]).T
copterCenterHeight = 0.15
cameraPosForUav = [0.03, 0, 0]
pic_w = 640
pic_h = 480
focal = 320
px = pic_w/2
py = pic_h/2

#
# ==================== 可视化开关（本实验专用） ====================
SHOW_9PTS = True       # True: 在预览窗口中画出目标无人机的9个采样点及其连线；False: 恢复原始的红框预览
SHOW_9PTS_DETAIL = True  # True: 额外画出 8 顶点围成的长方体棱边；False: 只画 9 个点、不连线
PRINT_9PTS = True      # True: 每个采样周期在终端打印9点的像素坐标
SHOW_9PTS_PERIOD = 5   # 每多少个采样周期打印/绘制一次（与主循环 num % 5 的节拍保持一致）
# ================================================================

# 9个点对应机体坐标系下的固定偏移量（单位 m），顺序与 getUav9Point() 的返回值严格一一对应。
# 这组数值就是"目标无人机几何外形"的全部先验：改机型时必须同步修改这里。
POINT_OFFSETS = [
    [-0.035, -0.035, -0.135 - 0.015],   # 0: z=-0.15  最低点（起落架/机腹）
    [0.245, -0.245, -0.035],            # 1: 右前顶点
    [0.245, 0.245, -0.035],             # 2: 右后顶点
    [-0.245, -0.245, -0.035],           # 3: 左前顶点
    [-0.245, 0.245, -0.035],            # 4: 左后顶点
    [0.13, -0.13, 0.17],                # 5: 右前上方顶点（桨盘以上）
    [0.13, 0.13, 0.17],                 # 6: 右后上方顶点
    [-0.13, -0.13, 0.17],               # 7: 左前上方顶点
    [-0.13, 0.13, 0.17],                # 8: 左后上方顶点
]
# 顶点索引（8个顶点）+ 中心点索引
POINT_CENTER_INDEX = 8
# 8个顶点围成长方体的12条棱（用索引表示），用于画出立体框
BOX_EDGES = [
    (1, 2), (2, 4), (4, 3), (3, 1),   # 下方四条棱（含点1~4，均在 z=-0.035 平面）
    (5, 6), (6, 8), (8, 7), (7, 5),   # 上方四条棱（含点5~8，均在 z=+0.17 平面）
    (1, 5), (2, 6), (3, 7), (4, 8),   # 连接上下的四条竖直棱
]
# 图例文字（点0与点1~4不在同一高度，见注释）
POINT_LEGEND = 'idx0=v-lowest  blue=idx1-4  red=idx5-8  yellow=center  * = out of frame'

# 打印9点在机体坐标系中的定义，方便对照机型几何参数
if PRINT_9PTS:
    print('=== 目标机 9 个采样点（相对几何中心的机体坐标，单位 cm）===')
    print('    说明：copterCenterHeight = %.2f m 表示"几何中心"位于 sendUE4Pos '
          '指令位置下方多远' % copterCenterHeight)
    for k, off in enumerate(POINT_OFFSETS):
        tag = 'C(中心)' if k == POINT_CENTER_INDEX else str(k)
        print('    %-8s x=%+7.2f  y=%+7.2f  z=%+7.2f' %
              (tag, off[0] * 100, off[1] * 100, off[2] * 100))
    print('    包络盒尺寸：X %.3f m  Y %.3f m  Z %.3f m' %
          (max(o[0] for o in POINT_OFFSETS) - min(o[0] for o in POINT_OFFSETS),
           max(o[1] for o in POINT_OFFSETS) - min(o[1] for o in POINT_OFFSETS),
           max(o[2] for o in POINT_OFFSETS) - min(o[2] for o in POINT_OFFSETS)))

# 创建一个独立窗口，用于放大观察9个点（不占用数据集生成窗口）
if SHOW_9PTS:
    cv2.namedWindow("9Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("9Points", 960, 720)  # 放大显示，取图分辨率仍由Config.json决定(640x480)

time.sleep(1)


def project_point(world_point, uav_pos, uav_ang):
    '''
    把世界系下的一个点投影为图像像素坐标（与主循环原逻辑完全一致）
    '''
    relative_pos = np.dot(eul2rot(uav_ang), np.transpose(
        relative_position(world_point, uav_pos)))
    return get_pic_situation(np.transpose(relative_pos), px, py, focal)


def body2world(offset, uav_pos, uav_ang):
    '''
    机体坐标系偏移量 -> 世界坐标。与 getUav9Point() 内部使用的是同一个变换，
    因此可用于反查每个点在机体坐标系中的真实位置。
    '''
    return np.array(uav_pos) + [0, 0, -copterCenterHeight] + \
        np.transpose(np.dot(np.linalg.inv(eul2rot(uav_ang)), np.transpose(np.array(offset))))


def draw_9_points(img, points, detail=True, center_index=POINT_CENTER_INDEX):
    '''
    在一张图上画出9个点：点0=青色(最低点)、点1~4=蓝色(下方顶点)、
    点5~8=红色(上方顶点)、几何中心=黄色。可选画出8个顶点围成的长方体棱边。
    '''
    out = img.copy()
    pts = [(int(round(p[0])), int(round(p[1]))) for p in points]

    if detail:
        # 先画棱边，再画点，避免线压住点
        for i, j in BOX_EDGES:
            cv2.line(out, pts[i], pts[j], (0, 255, 0), 1, cv2.LINE_AA)

    for i, p in enumerate(pts):
        if i == center_index:
            color, label = (0, 255, 255), 'C'      # 8 几何中心：黄
        elif i == 0:
            color, label = (255, 255, 0), str(i)   # 0 最低点：青
        elif i <= 4:
            color, label = (255, 0, 0), str(i)     # 1~4 下方顶点：蓝
        else:
            color, label = (0, 0, 255), str(i)     # 5~8 上方顶点：红

        # 该点是否落在图像内（用于辅助判断框被裁掉的原因）
        inside = (0 <= p[0] < pic_w) and (0 <= p[1] < pic_h)

        cv2.circle(out, p, 4, color, -1, cv2.LINE_AA)   # 实心圆点
        cv2.circle(out, p, 6, (255, 255, 255), 1, cv2.LINE_AA)  # 白色外圈，便于在复杂背景下辨认

        tag = label if inside else label + '*'          # 带 * 表示该点投影到画幅之外
        org = (p[0] + 8, p[1] - 8)
        cv2.putText(out, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.putText(out, POINT_LEGEND,
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, POINT_LEGEND,
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def eul2rot(theta):
    '''
    欧拉角转旋转矩阵，与世界坐标系方向、欧拉角旋转顺序、旋转正方向的定义有关
    '''
    R_x = np.array([[1,         0,                  0],
                    [0,         math.cos(theta[0]), math.sin(theta[0])],
                    [0,         -math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      -math.sin(theta[1])],
                    [0,                     1,      0],
                    [math.sin(theta[1]),   0,      math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]),    math.sin(theta[2]),    0],
                    [-math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

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


def getUav9Point(uavPos, copterCenterHeight, uavAng, UAVh=0.185, UAVw=0.185, eurTheta=[0, 0, 0]):
    '''
    通过输入UAV的坐标以及相关信息，获取九个空间点的坐标
    '''
    # res = []
    center = np.array(uavPos)+[0, 0, -copterCenterHeight]
    return [

        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([-0.035, -0.035, -0.135-0.015])))),
        # 需要把机翼也要包含在框内
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([0.245, -0.245, -0.035])))),
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([0.245, 0.245, -0.035])))),
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([-0.245, -0.245, -0.035])))),
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([-0.245, 0.245, -0.035])))),

        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([0.13, -0.13, 0.17])))),
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([0.13, 0.13, 0.17])))),
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([-0.13, -0.13, 0.17])))),
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                              np.transpose(np.array([-0.13, 0.13, 0.17]))))

    ]


def get_image_size(knownWidth, focalLength, distance):

    return (knownWidth * focalLength) / distance


def get_pic_situation(relative_pos, px, py, focalLength):
    fx = (focalLength/relative_pos[0]) * \
        np.linalg.norm([relative_pos[0], relative_pos[2]])
    fy = (focalLength/relative_pos[0]) * \
        np.linalg.norm([relative_pos[0], relative_pos[1]])
    cx = get_image_size(relative_pos[1], fx, np.linalg.norm(
        [relative_pos[0], relative_pos[2]])) + px
    cy = get_image_size(relative_pos[2], fy, np.linalg.norm(
        [relative_pos[0], relative_pos[1]])) + py

    return [round(cx), round(cy)]


def relative_position(sit_a, sit_b):
    a = [sit_a[0] - sit_b[0], sit_a[1] - sit_b[1], sit_a[2] - sit_b[2]]
    return a


def get_distance(relative_pos):
    distance = np.linalg.norm(relative_pos)
    return distance


def xyxy2xywh(x1, x2):
    cent = [(x1[0]+x2[0])/2, (x1[1]+x2[1])/2]
    wh = [x2[0]-x1[0], x2[1]-x1[1]]
    return cent, wh


def xywh2yolo(c, wh):
    y1 = round(c[0]/pic_w, 8)
    y2 = round(c[1]/pic_h, 8)
    y3 = round(wh[0]/pic_w, 8)
    y4 = round(wh[1]/pic_h, 8)
    return y1, y2, y3, y4


# 创建一个txt文件，文件名为mytxtfile,并向文件写入msg
def text_create(name, msg):
    desktop_path = labels  # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.write(msg)  # msg也就是下面的Hello world!
    file.close()


# ==================== 数据集保存位置配置（本实验专用） ====================
# 数据集根目录，支持四种配置方式（优先级从高到低）：
#   1) 命令行参数： python test_dateset.py --save-dir "D:/datasets/SO3"
#   2) 环境变量：   set SO3_DATASET_ROOT=D:\datasets\SO3          (Windows CMD)
#                   $env:SO3_DATASET_ROOT="D:\datasets\SO3"      (PowerShell)
#   3) Config.json 中的 dataset 段（推荐，见 Config.json 内的注释）
#   4) 直接修改下面的默认值
# 取值说明：
#   ''         -> 使用脚本所在目录（与原脚本行为一致）
#   绝对路径   -> 直接按该路径保存
#   相对路径   -> 相对“脚本所在目录”解析
# 注意：Config.json 用 // 和 # 作注释，因此其中的路径不要包含 "//"（会被当作注释截断）
DATASET_ROOT = ''
# 是否在根目录下再套一层时间戳子目录（%Y%m%d_%H%M%S）：
#   True  -> <DATASET_ROOT>/20240101_120000/{images,labels}，多次运行互不覆盖
#   False -> <DATASET_ROOT>/{images,labels}，多次运行写入同一目录
USE_TIMESTAMP_SUBDIR = True
# 图/标签子目录名（标准 YOLO 布局为 images/labels，可按需改名）
IMAGE_SUBDIR = 'images'
LABEL_SUBDIR = 'labels'
# 配置文件（与 VisionCaptureApi.jsonLoad 使用的是同一个文件），可用 --config 指定其他文件
CONFIG_FILE = 'Config.json'
# ========================================================================


def parse_args():
    '''
    解析命令行参数；用 parse_known_args 兼容 RflySim 等外部调用时附带的额外参数
    '''
    parser = argparse.ArgumentParser(
        description='生成 SO3 视觉数据集（保存位置可配置）')
    parser.add_argument('--save-dir', '--save_dir', dest='save_dir', default=None,
                        help='数据集保存根目录，优先于环境变量 SO3_DATASET_ROOT、Config.json 和 DATASET_ROOT')
    parser.add_argument('--config', dest='config', default=None,
                        help='配置文件路径（默认脚本目录下的 Config.json），其中的 dataset 段用于配置保存位置')
    parser.add_argument('--timestamp', dest='timestamp', action='store_true',
                        default=None, help='在根目录下创建时间戳子目录（默认行为）')
    parser.add_argument('--no-timestamp', '--no_timestamp', dest='timestamp',
                        action='store_false', default=None,
                        help='不创建时间戳子目录，直接写入 <根目录>/images 与 <根目录>/labels')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('[提示] 已忽略无法识别的命令行参数: {}'.format(unknown))
    return args


def load_dataset_config(config_path):
    '''
    读取 Config.json 中的 dataset 配置段。
    解析方式与 VisionCaptureApi.jsonLoad 保持一致：先过滤 // 与 # 注释，再交给 json 解析。
    读取或解析失败时返回空 dict，不影响脚本主流程。
    '''
    if not config_path or not os.path.isfile(config_path):
        print('[配置] 未找到 {}，跳过其中的 dataset 配置'.format(config_path))
        return {}
    try:
        lines = []
        with open(config_path, 'r', encoding='utf-8') as f:
            for row in f.readlines():
                if row.strip().startswith('//') or row.strip().startswith('#'):
                    continue
                row = re.sub('//.*', '', row)   # 去掉 // 注释
                row = re.sub('#.*', '', row)    # 去掉 # 注释
                lines.append(row)
        js_data = json.loads('\n'.join(lines))
    except Exception as err:
        print('[警告] 解析 {} 失败({})，将忽略其中的 dataset 配置'.format(config_path, err))
        return {}
    if not isinstance(js_data, dict):
        print('[警告] {} 的顶层不是 JSON 对象，将忽略其中的 dataset 配置'.format(config_path))
        return {}
    ds = js_data.get('dataset', {})
    if isinstance(ds, list):        # 兼容 "dataset":[ {...}, {...} ] 的写法，后面的覆盖前面的
        merged = {}
        for item in ds:
            if isinstance(item, dict):
                merged.update(item)
        ds = merged
    if not isinstance(ds, dict):
        print('[警告] Config.json 的 dataset 段应为对象或对象数组，当前为 {}，已忽略'.format(
            type(ds).__name__))
        return {}
    # 键名统一为小写并去掉 _ - 空格，从而容忍 SaveDir / save_dir / save-dir 等写法
    return {re.sub(r'[_\-\s]', '', str(k)).lower(): v for k, v in ds.items()}


def cfg_get_str(cfg, *names):
    '''
    从（已归一化键名的）dataset 配置中取第一个非空字符串
    '''
    for name in names:
        value = cfg.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def cfg_get_bool(cfg, *names):
    '''
    从（已归一化键名的）dataset 配置中取布尔值，兼容 true/false、1/0、"true"/"false"
    '''
    for name in names:
        value = cfg.get(name)
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


def pick_first(*candidates):
    '''
    按顺序返回第一个非空的候选值，返回 (来源说明, 取值)
    '''
    for source, value in candidates:
        if value is not None and value != '':
            return source, value
    return candidates[-1]


def resolve_save_root(cli_save_dir, env_save_dir, cfg_save_dir, script_dir):
    '''
    按 命令行 > 环境变量 > Config.json > 脚本内常量 的优先级确定数据集根目录，
    并把相对路径统一解析为绝对路径
    '''
    source, root = pick_first(
        ('命令行参数 --save-dir', cli_save_dir),
        ('环境变量 SO3_DATASET_ROOT', env_save_dir),
        ('Config.json 的 dataset.SaveDir', cfg_save_dir),
        ('脚本内 DATASET_ROOT', DATASET_ROOT),
        ('脚本所在目录（默认）', script_dir))
    # 展开 ~ 与 %VAR% / $VAR% 形式的环境变量
    root = os.path.expanduser(os.path.expandvars(root))
    if not os.path.isabs(root):
        root = os.path.join(script_dir, root)   # 相对路径按脚本所在目录解析
    print('[配置] 数据集保存根目录来源: {}'.format(source))
    return os.path.normpath(root)


# 以当前日期和时间创建文件夹，准备写入新图片
path_prefix = sys.path[0] or os.path.dirname(os.path.abspath(__file__))  # 脚本所在路径
_args = parse_args()

# 1) 读取 Config.json（其中的 dataset 段用于配置保存位置）
config_path = _args.config or CONFIG_FILE
if not os.path.isabs(config_path):
    config_path = os.path.join(path_prefix, config_path)   # 相对路径按脚本所在目录解析
config_path = os.path.normpath(config_path)
_dataset_cfg = load_dataset_config(config_path)

# 2) 保存根目录：命令行 > 环境变量 > Config.json > 脚本内常量 > 脚本目录
save_root = resolve_save_root(_args.save_dir, os.environ.get('SO3_DATASET_ROOT'),
                              cfg_get_str(_dataset_cfg, 'savedir', 'savepath',
                                          'datasetroot', 'savedirpath'),
                              path_prefix)

# 3) 是否套一层时间戳子目录：命令行 > Config.json > 脚本内常量
_cfg_timestamp = cfg_get_bool(_dataset_cfg, 'usetimestampsubdir', 'usetimestamp', 'timestamp')
use_timestamp = _args.timestamp if _args.timestamp is not None else (
    _cfg_timestamp if _cfg_timestamp is not None else USE_TIMESTAMP_SUBDIR)

# 4) 图/标签子目录名：Config.json > 脚本内常量
image_subdir = cfg_get_str(_dataset_cfg, 'imagesubdir', 'imgsubdir', 'imagesdir') or IMAGE_SUBDIR
label_subdir = cfg_get_str(_dataset_cfg, 'labelsubdir', 'labsubdir', 'labelsdir') or LABEL_SUBDIR

path_dir = os.path.join(save_root, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) \
    if use_timestamp else save_root
os.makedirs(path_dir, exist_ok=True)
print("config.json: {}".format(config_path))
print("save_root: {}".format(save_root))
print("path_dir: {}".format(path_dir))
print("use_timestamp: {}".format(use_timestamp))
path_img = os.path.join(path_dir, image_subdir)
labels = os.path.join(path_dir, label_subdir)
os.makedirs(path_img, exist_ok=True)
os.makedirs(labels, exist_ok=True)


print(path_img)

startTime = time.time()
lastTime = time.time()
timeInterval = 0.1
cnt = 0
num = 0
shape_num = 9  # 目标自凸点个数以及自身原点,当前目标是UAV,使用box包裹，那就是有8个顶点和一个原点
pic_sit = [[0 for col in range(2)] for row in range(shape_num)]

camere_pos = PosInit + \
    np.transpose(np.dot(np.linalg.inv(
        eul2rot(angCopterE)), np.transpose(cameraPosForUav)))  # 观测飞机的相机位姿是固定的


while True:
    lastTime = lastTime + timeInterval
    sleepTime = lastTime - time.time()
    if sleepTime > 0:
        time.sleep(sleepTime)
    else:
        lastTime = time.time()

    num = num+1
    # Call every 0.5 seconds
    if num % 5 == 0:
        TargePos = [InitTargePos[0]+random.randint(0, 200)/100.0, InitTargePos[1] +
                    random.randint(-60, 60)/100.0, InitTargePos[2]+random.randint(-100, 100)/200.0]
        TargeAng = [InitTargeAng[0]+random.randint(-50, 50)/50.0*20/180.0*math.pi, InitTargeAng[1]+random.randint(
            -50, 50)/50.0*20/180.0*math.pi, InitTargeAng[2]+random.randint(-50, 50)/50.0*30/180.0*math.pi]
        # TargePos = [2,1,-8.086-4.8]
        # TargeAng = [0,-np.pi/6,0]

        ue.sendUE4Pos(100, 1, 0, TargePos, TargeAng, -1)
        points = getUav9Point(TargePos, copterCenterHeight, TargeAng)
        time.sleep(0.2)
        i = 0

        if vis.hasData[0]:
            # The following code will be executed per 3s
            img = vis.Img[0]
            cv2.imshow("pic1", img)  # Show the processed image
            cv2.waitKey(1)

        for t_point in points:

            # camere_pos = PosInit
            pos_p = project_point(t_point, camere_pos, angCopterE)
            pic_sit[i] = pos_p
            i = i+1

        pic_sit = np.array(pic_sit)

        # 打印9点的像素坐标，便于核对每个点落在画面的哪个位置
        if PRINT_9PTS:
            names = ['0(最低点)', '1(右前)', '2(右后)', '3(左前)',
                     '4(左后)', '5(右上)', '6(右后上)', '7(左上)', 'C(中心)']
            print('--- 9点像素坐标 (cx, cy)，画幅 %dx%d ---' % (pic_w, pic_h))
            for k in range(shape_num):
                print('  %-10s -> (%4d, %4d)' %
                      (names[k], pic_sit[k, 0], pic_sit[k, 1]))

        #  取bbox左上和右下的点
        x1y1 = [min(pic_sit[:, 0]), min(pic_sit[:, 1])]
        x2y2 = [max(pic_sit[:, 0]), max(pic_sit[:, 1])]

        if (x1y1[0]+x2y2[0])/2 > 0 and (x1y1[1]+x2y2[1])/2 > 0 and (x1y1[0]+x2y2[0])/2 < pic_w and (x1y1[1]+x2y2[1])/2 < pic_h:
            target = 1
        else:
            target = 0

        if target == 1 and x1y1[0] < 0:
            x1y1[0] = 0
        if target == 1 and x1y1[1] < 0:
            x1y1[1] = 0
        if target == 1 and x2y2[0] > pic_w:
            x2y2[0] = pic_w
        if target == 1 and x2y2[1] > pic_h:
            x2y2[1] = pic_h

        if vis.hasData[0]:
            # The following code will be executed per 3s
            img1 = vis.Img[0]
            # Get the first camera view and change to gray
            # 可是化计算得到的框，验证是否正确
            img_show = vis.Img[0].copy()
            cv2.rectangle(img_show, (int(x1y1[0]), int(x1y1[1])),
                          (int(x2y2[0]), int(x2y2[1])), (0, 0, 255))

            # ---- 画出9个采样点及其空间连线（新增可视化） ----
            if SHOW_9PTS:
                img_show = draw_9_points(
                    img_show, pic_sit, detail=SHOW_9PTS_DETAIL)
                # 用最近邻放大，避免点位/连线被插值糊掉
                cv2.imshow("9Points", cv2.resize(
                    img_show, (960, 720), interpolation=cv2.INTER_NEAREST))

            cv2.imshow("pic1", img_show)  # Show the processed image
            if target == 1:
                # cv2.imwrite(os.path.join(path_img, "{}.jpg".format(cnt)), img1)
                cv2.imwrite(path_img + '/' + "{}.jpg".format(cnt), img1)
                cent, wh = xyxy2xywh(x1y1, x2y2)
                y1, y2, y3, y4 = xywh2yolo(cent, wh)
                desktop_path = labels  # 新创建的txt文件的存放路径
                full_path = desktop_path + '/' + \
                    '{}.txt'.format(cnt)  # 也可以创建一个.doc的word文档
                file = open(full_path, 'w')
                file.writelines(['0', ' ', str(y1), ' ', str(
                    y2), ' ', str(y3), ' ', str(y4)])  # YOLO格式生成
                file.close()

        # add image storing algorithm for MATLAB offline calibration
        # or add online calibration algorithm here

        cnt += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

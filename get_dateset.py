# import required libraries
import random
import os
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
ue.sendUE4Pos(100, 3, 0, InitTargePos, InitTargeAng)
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

time.sleep(1)


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


# 以当前日期和时间创建文件夹，准备写入新图片
path_prefix = sys.path[0]    # 当前工作路径
path_dir = os.path.join(
    path_prefix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(path_dir)
print("path_dir: {}".format(path_dir))
path_img = os.path.join(path_dir, "images")
labels = os.path.join(path_dir, "labels")
os.makedirs(path_img)
os.makedirs(labels)


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

        ue.sendUE4Pos(100, 3, 0, TargePos, TargeAng, -1)
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
            relative_pos = np.dot(eul2rot(angCopterE), np.transpose(
                relative_position(t_point, camere_pos)))
            pos_p = get_pic_situation(
                np.transpose(relative_pos), px, py, focal)
            pic_sit[i] = pos_p
            i = i+1
            print(pos_p)

        pic_sit = np.array(pic_sit)

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

import os
import random

'''
对图片数据集进行随机分类
以8: 1: 1的比例分为训练数据集，验证数据集和测试数据集
运行后在ImageSets文件夹中会出现四个文件
'''
ROOT = 'E:/Project/Python/4-GenVisionDataSet/20220706_080343/'  # 根据自己的目标进行替换
train_percent = 0.9
txtfilepath = ROOT + 'labels'
txtsavepath = ROOT + 'ImageSets'
datasavepath = ROOT + 'data'
# 获取该路径下所有文件的名称，存放在list中
total_txt = os.listdir(txtfilepath)

num = len(total_txt)
list = range(num)
tr = int(num * train_percent)
train = random.sample(list, tr)

if not os.path.exists(ROOT + 'ImageSets/'):
    os.makedirs(ROOT + 'ImageSets/')

if not os.path.exists(ROOT + 'data/'):
    os.makedirs(ROOT + 'data/')

ftrain = open(ROOT + 'ImageSets/train.txt', 'w')
fval = open(ROOT + 'ImageSets/val.txt', 'w')
dtrain = open(ROOT + 'data/train.txt', 'w')
dval = open(ROOT + 'data/val.txt', 'w')

for i in list:
    # 获取文件名称中.xml之前的序号
    name = total_txt[i][:-4]
    if i in train:
        ftrain.write(name+'\n')
        dtrain.writelines([str(ROOT), 'images/', str(i), '.jpg', '\n'])
    else:
        fval.write(name+'\n')
        dval.writelines([str(ROOT), 'images/', str(i),
                        '.jpg'.format(name), '\n'])

ftrain.close()
fval.close()

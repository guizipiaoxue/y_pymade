#--------------------对数据集的标注进行操作---------------------------

import os
from xml.etree import ElementTree as ET
import random

#---------------定义参数----------------

# 所有类
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# xml文件路径
Annotations_path = '../data/VOCdevkit/VOC2012/Annotations/'

# 所有的xml文件
xml_files = os.listdir(Annotations_path)

# 打乱数据集
random.shuffle(xml_files)

# 训练集，测试集按照 8:2来分割
train_num = int(len(xml_files) * 0.8)
# 训练集的名字列表
train_files = xml_files[:train_num]
# 测试集的名字列表
test_files = xml_files[train_num:]

# 测试集以及训练集的储存路径
train_set_path = './voc_train.txt'
test_set_path = './voc_test.txt'


#-------------------解析xml文件函数----------------------

def parse_rec(filename):
    tree = ET.parse(filename)
    # 一个图片里面有多个object，故这里设置一个列表装下所有标记的物体
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        # xml文件里面有difficult属性，如果是difficult就跳过
        difficult = int(obj.find('difficult').text)
        if difficult != 0:
            continue
        # 保留类的名字，框的大小位置
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(eval(bbox.find('xmin').text)),
                              int(eval(bbox.find('ymin').text)),
                              int(eval(bbox.find('xmax').text)),
                              int(eval(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects

#-----------------------写入文件---------------------------

def write_txt(file_list, set_path):
    count = 0
    with open(set_path, 'w') as f:
        for xml_file in file_list:
            count += 1
            # 获取图片的名字
            img_path = xml_file.split('.')[0] + '.jpg'

            # 获取相关信息
            result = parse_rec(Annotations_path + xml_file)

            if len(result) == 0:
                print(xml_file)
                continue

            # 先写名字
            f.write(img_path)

            for obj in result:
                class_name = obj['name']
                bbox = obj['bbox']
                class_name = VOC_CLASSES.index(class_name)
                f.write(f' {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {class_name}')
            f.write('\n')

#-----------------运行文件--------------------
if __name__ == '__main__':
    write_txt(train_files, train_set_path)
    write_txt(test_files, test_set_path)
















#----------------------构建数据加载器类------------------------

import os
import numpy as np
import torch
import random
import torchvision.transforms as T
from torch.utils.data import Dataset
import cv2


#-----------------数据加载器------------------------
class YoloDataset(Dataset):
    """
    数据迭代器类
    """
    # 设置图片的大小
    image_size = (448, 448)

    def __init__(self, root, list_file, train=True, transform=None ):
        """
        初始化类参数
        :param root: 图片的根目录
        :param list_file: 图片信息
        :param train: 是否是训练集
        :param transform: 预处理
        """
        super(YoloDataset, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.files_name = [] # 储存图片名字
        self.boxes = [] # 储存边框
        self.labels = [] # 储存标签
        self.mean = (123, 117, 104) # RGB归一化

        # 打开读取txt文件
        with open(list_file, 'r') as f:
            lines = f.readlines()

        # 每一行
        for line in lines:
            imformations = line.strip().split()
            # 保存名字
            self.files_name.append(imformations[0])


            # 计算当前行包含的对象数量，每五个数据表示一个对象
            num_boxes = (len(imformations) - 1) // 5

            # 初始化当前图片的边界框和标签列表
            box = []
            label = []

            # 遍历当前行的所有对象
            for i in range(num_boxes):
                # 读取边界框的坐标信息
                x = float(imformations[1 + 5 * i])
                y = float(imformations[2 + 5 * i])
                x2 = float(imformations[3 + 5 * i])
                y2 = float(imformations[4 + 5 * i])

                # 读取对象的类别标签
                c = imformations[5 + 5 * i]

                # 将边界框坐标添加到列表中，并转换为Tensor格式
                box.append([x, y, x2, y2])

                # 将标签转换为整数并加1（因为标签通常从1开始计数），然后添加到列表中
                label.append(int(c) + 1)

            # 将当前图片的边界框和标签信息转换为LongTensor格式添加到对应的列表中
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

        # 记录数据集中的样本数量
        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # 索引名字得到文件名
        file_name = self.files_name[index]

        # 获取图片，标签，边框
        img = cv2.imread(os.path.join(self.root, file_name))
        boxes = self.boxes[index]
        labels = self.labels[index]

        #------------------这里可以添加数据增强函数--------------------------
        #------------------但是我不想写了，自行去了解，我提供接口--------------

        # if self.train:
        #   ........

        # 归一化这个框的位置
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)

        # cv2换通道
        img = self.BGR2RBG(img)
        img = cv2.resize(img, self.image_size)

        # 相量编码转换 *important
        target = self.encoder(boxes, labels)

        for t in self.transform:
            img = t(img)

        return img, target


    #------------------------编码-----------------------------
    def encoder(self, boxes, labels):
        """
        注意，这个函数是对一张图片进行处理，里面可能有多个目标，多个框
        将边界框和标签编码为我的目标张量
        :param boxes:
        :param labels:
        :return: target( 7 * 7 * 30 )
        这里为什么是这个大小
        1. 我们把图片切割成7 * 7 当然你可以自己转换
        2. 我们有20个类， 加上俩层的框和置信度，为 20 + 5 + 5 = 30
        """

        # 设定网格数量
        grid_num = 7
        cell_size = 1. / grid_num

        # 设置空张量
        target = torch.zeros((grid_num, grid_num, 30))

        # 计算这个框的高度，宽度，中心点位置
        # wh是一个tensor，[[w, h], [w, h] ......]
        # 切片自己复习
        wh = boxes[:, 2:] - boxes[:, :2]

        # 中心坐标
        # cxcy是一个tensor [[cx, cy], [cx, cy] ......]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

        # 遍历所有的中心点
        for i in range(cxcy.size()[0]):

            # 获取坐标
            cxcy_sample = cxcy[i]

            # 计算中心点所在的索引，就是这个中心坐标在这7个网格中的哪一个网格
            cxcy_index_cell = (cxcy_sample / cell_size).ceil() - 1

            # 至于这里为什么是1再是0，自行复习numpy切片操作
            # 这里通道4，9被设置为了置信度通道，有物体的则标注为1
            target[cxcy_index_cell[1], cxcy_index_cell[0], 4] = 1
            target[cxcy_index_cell[1], cxcy_index_cell[0], 9] = 1

            # 计算中心点在网格里面的相对坐标，所在网格的左上角
            cxcy_realate_position = cxcy_index_cell * cell_size



            # 填充目标张量的网格的宽度，高度等信息
            # 这里有算法的不严谨性，其中如果有物体重复了中心点呢？这里的target就会被置换
            # 解决方法：切割多份等等
            target[cxcy_index_cell[1], cxcy_index_cell[0], 2:4] = wh[i]

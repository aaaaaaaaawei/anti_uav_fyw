from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC']


class Net(nn.Module): # 定义了一个名为 Net 的神经网络模型

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone # 骨干网络，进行特征提取 使用的是alexnet
        self.head = head # 头部网络
    #接收两个输入张量 z 和 x，分别表示目标样本和搜索图像。
    #首先将目标样本和搜索图像分别通过骨干网络（backbone）进行特征提取
    #然后将提取到的特征传递给头部网络（head）进行后续处理，最终返回头部网络的输出结果
    def forward(self, z, x):#跟踪推理阶段用不到 训练的时候用得到
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)#参数注入

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()#判断gpu是否可用
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')#设定运行时的设备  此处我的配置有问题 self.cuda为false，没法用gpu去跑

        # setup model
        self.net = Net( # 初始化整个网络，包括骨干网络和head
            backbone=AlexNetV1(), # 骨干网络
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)#初始化网络层的参数
        
        # load checkpoint if provided
        if net_path is not None: #判断预训练模型是否存在
            self.net.load_state_dict(torch.load( #加载预训练模型
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device) # net迁移到device上，从cpu迁移到gpu

        # setup criterion
        self.criterion = BalancedLoss()# 定义损失函数 测试阶段不用

        # setup optimizer
        self.optimizer = optim.SGD( # 定义SGD优化器
            self.net.parameters(),
            lr=self.cfg.initial_lr, # 初始学习率
            weight_decay=self.cfg.weight_decay,  # 权重衰减 防止过拟合
            momentum=self.cfg.momentum)  # 动量 加速模型收敛
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)  #学习率调整策略

    def parse_args(self, **kwargs):
        # default parameters 默认参数
        cfg = {
            # basic parameters
            'out_scale': 0.001,#输出特征图的缩放比例
            'exemplar_sz': 127,#样本图像的大小，通常是目标图像的一部分
            'instance_sz': 255,#搜索区域的大小，即在图像中搜索目标的区域大小
            'context': 0.5,#上下文比例，用于确定搜索区域相对于样本图像的大小
            # inference parameters 推理阶段 测试阶段
            'scale_num': 3,#尺度数，用于表示在多个尺度上搜索目标
            'scale_step': 1.0375,#尺度步长，用于确定在每个尺度上的搜索间隔
            'scale_lr': 0.59,#尺度学习率，用于更新尺度参数
            'scale_penalty': 0.9745,# 尺度惩罚，用于惩罚尺度变化
            'window_influence': 0.176,#余弦窗的影响参数，用于调整响应图的平滑度
            'response_sz': 17,#响应图的大小
            'response_up': 16,#响应图上采样的倍数
            'total_stride': 8,#总步长，用于确定特征图的步长
            # train parameters
            'epoch_num': 80, #训练的 epoch 数量，即训练数据集将被遍历的次数 我修改原始为50
            # 'epoch_num': 50, #原始数据
            # 'batch_size': 8,#每个训练批次中的样本数量
            'batch_size': 16,  # 每个训练批次中的样本数量
            # 'num_workers': 4,  # 用于加载数据的线程数量
             'num_workers': 8,  # 用于加载数据的线程数量
            # 'num_workers': 32,#用于加载数据的线程数量
            'initial_lr': 1e-2,#初始学习率
            'ultimate_lr': 1e-5,#最终学习率
            'weight_decay': 5e-4,#权重衰减（L2 正则化）的参数，用于控制模型的复杂度
            'momentum': 0.9,#动量参数，用于控制随机梯度下降SGD算法的更新方向
            'r_pos': 16, #'r_pos' 和 'r_neg'表示正样本和负样本的半径大小，用于目标检测任务中的样本采样。
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32) #将bbox从[l,t,w,h]转为[y,x,h,w],裁剪图像使追踪目标位于图像正中间
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz#计算上采样size 用于将17*17*1还原成272*272
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz)) # 初始化汉明窗
        self.hann_window /= self.hann_window.sum()# 对汉宁窗进行归一化，使其总和为1，以确保加权后的结果保持一致性

        # search scale factors尺度因子
        self.scale_factors = self.cfg.scale_step ** np.linspace(#通过线性插值计算search scale factors
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))#样本图像patch的size
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz #搜索图像patch的size
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))#计算三颜色通道的均值，用于填充
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color) #获取样本图像的patch
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float() #将patch的channel放在最前，并在channel前加一个1维 1*3*127*127
        self.kernel = self.net.backbone(z) #将patch送入backbone 输出的feature作为互相关操作时的卷积核kernel
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors] #对每一个scale_factors裁剪出search patch
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        """
        search patch和examplar patch的区别是，examplar裁剪后返回的是单张image，需要用squeeze手动补1个维度才能送入backbone
        而search patch需要对每一个scale_fatcors裁剪出一个对应的search_patch(这里是共3张)，返回的直接就是4维张量。
        这样search patch和examplar patch生成的特征做卷积后输出的responses其实就是3张堆叠在一起的特征图，对应3张search patch
        后续再通过np.amax和np.argmax找出峰值score最大的那张图作为最优响应图
        """
        # responses
        x = self.net.backbone(x) #将search patch送入backbone获得其特征向量 当前帧的特征
        responses = self.net.head(self.kernel, x) #用样本图像特征kernel和x做卷积，输出一个shape=[3，1，17，17]的
        responses = responses.squeeze(1).cpu().numpy()#[3，1，17，17]-》[3,17,17]

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize( #3*17*17->3*272*272
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses]) #上采样至272
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty #对发生形变的相应图进行惩罚 中间那张尺度因子为1不惩罚
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))#找到峰值score最大的响应图索引

        # peak location
        response = responses[scale_id] #最终的响应图 272*272
        response -= response.min()
        response /= response.sum() + 1e-16 #归一化
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window #余弦窗惩罚  self.cfg.window_influence为超参数，实验试出来的，0.176
        loc = np.unravel_index(response.argmax(), response.shape)#loc为峰值点坐标

        # locate target center 将17*17还原
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2 #峰值点与response中心之间的位移 目标物相对于上一帧之间的偏移
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up #倒推response上的位移映射在patch上的位移
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz #倒推计算在image上的位移
        self.center += disp_in_image #修正center 更新后的目标位置(此时目标还在图片中间)

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id] #线性插值更新scale
        self.target_sz *= scale #更新target_sz
        self.z_sz *= scale #不更新也可以没用到
        self.x_sz *= scale # 更新搜索区域size

        # return 1-indexed and left-top based bounding box  由中心点center，宽高转换为左上角，宽高
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files) # 当前序列视频帧总数
        boxes = np.zeros((frame_num, 4))#初始化bbox ndarry
        boxes[0] = box #设置为第一帧ground truth
        times = np.zeros(frame_num) #初始化时间数组

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box) #将首帧信息固定下来作为后续帧的卷积核 第一帧使用init方法 box为首帧的groundtruth
            else:
                boxes[f, :] = self.update(img) # 非第一帧时调用update
            times[f] = time.time() - begin  # 计算当前帧跟踪时长

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained_anti_uav'):
        # set to train mode
        self.net.train()#训练模式

        # create save_dir folder
        if not os.path.exists(save_dir): #若 'pretrained'文件夹不存在，创建对应路径
            os.makedirs(save_dir)

        # setup dataset
        # 实例化一个SiamFCTransforms对象，用于对图像的处理
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        # siamFC在训练中是每个序列取一对图片，一张作为examplar，一张用作search
        # #两张图片间隔不超过T== 100.
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels

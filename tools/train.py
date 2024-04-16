from __future__ import absolute_import

import os
from got10k.datasets import *

from siamfc import TrackerSiamFC

# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())


if __name__ == '__main__':
    # root_dir = os.path.expanduser('E:\\goole_download\\3rd_Anti-UAV_train_val')
    root_dir = os.path.expanduser('D:\\Study\\01_Deep_Learning\\code\\siamfc-pytorch\\data\\GOT-10K')
    #初始化一个got10k的数据，完成了数据的完整性校验，获取序列名和对应的groundtruth.txt文件存入list中
    # seqs = GOT10k(root_dir, subset='track1_test', return_meta=True)
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    # 初始化一个tracker（未加载预训练模型）
    tracker = TrackerSiamFC()
    #开始训练
    tracker.train_over(seqs)

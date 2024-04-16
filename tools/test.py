from __future__ import absolute_import

import os
from got10k.experiments import *

import sys

# print('fyw/n')
sys.path.append('D:\\Study\\01_Deep_Learning\\code\\siamfc-pytorch')
sys.path.append('D:\\Study\\01_Deep_Learning\\code\\siamfc-pytorch\\data')
# sys.path.append('D:\\Study\\01_Deep_Learning\\code\\siamfc-pytorch\\tools\\pretrained\\siamfc_alexnet_e10.pth')
# print(sys.path)

from siamfc import TrackerSiamFC

#我加入
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    net_path = 'D:\Study\\01_Deep_Learning\code\siamfc-pytorch-anti_uav-updata\\tools\pretrained_anti_uav\\siamfc_alexnet_e30.pth'
    # net_path = 'pretrained/siamfc_alexnet_e10.pth'
    tracker = TrackerSiamFC(net_path=net_path)  # 网络

    root_dir = os.path.expanduser('E:\goole_download\\3rd_Anti-UAV_train_val\\train')
    # root_dir = os.path.expanduser('../data/OTB100')
    e = ExperimentOTB(root_dir, version='anti_uav')  # got10k提供的一个包
    e.run(tracker, visualize=True)
    e.report([tracker.name])

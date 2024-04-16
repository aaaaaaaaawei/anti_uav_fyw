import os
import shutil

source_dir = r'E:\goole_download\3rd_Anti-UAV_train_val\train\20190925_141417_1_7\img\img'
target_dir = r'E:\goole_download\3rd_Anti-UAV_train_val\train\20190925_141417_1_7\img'

# 遍历源目录下的所有文件
for filename in os.listdir(source_dir):
    # 构建源文件路径和目标文件路径
    source_file = os.path.join(source_dir, filename)
    target_file = os.path.join(target_dir, filename)
    # 将文件移动到目标文件夹中
    shutil.move(source_file, target_file)

os.rmdir(source_dir)
print("All files moved to the target directory.")

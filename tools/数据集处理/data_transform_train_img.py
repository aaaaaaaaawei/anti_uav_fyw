# import os
# import shutil
#
# # 源目录
# source_dir = r'E:\goole_download\3rd_Anti-UAV_train_val\train\01_1751_0250-1750'
# # 目标目录
# target_dir = os.path.join(source_dir, 'img')
#
# # 创建目标目录
# os.makedirs(target_dir, exist_ok=True)
#
# # 遍历源目录下的所有文件
# for filename in os.listdir(source_dir):
#     # 检查文件是否为图片文件
#     if filename.endswith(('.png', '.jpg', '.jpeg')):
#         # 构建源文件路径和目标文件路径
#         source_file = os.path.join(source_dir, filename)
#         target_file = os.path.join(target_dir, filename)
#         # 将图片文件移动到目标文件夹中
#         shutil.move(source_file, target_file)
#
# print("All images moved to the 'img' folder.")


import os
import shutil

# 源目录
source_parent_dir = r'E:\goole_download\3rd_Anti-UAV_train_val\train'

# 遍历源目录下的所有子目录
for subdir in os.listdir(source_parent_dir):
    source_dir = os.path.join(source_parent_dir, subdir)
    target_dir = os.path.join(source_dir, 'img')

    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    print(f"Created target directory: {target_dir}")

    # 遍历当前子目录下的所有文件
    for filename in os.listdir(source_dir):
        # 检查文件是否为图片文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 构建源文件路径和目标文件路径
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            print(f"Moving {source_file} to {target_file}")
            # 将图片文件移动到目标文件夹中
            shutil.move(source_file, target_file)

    print(f"All images moved to the 'img' folder in '{subdir}' directory.")




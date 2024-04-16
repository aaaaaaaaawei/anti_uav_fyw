import os

# 指定目录路径
directory = r'E:\goole_download\3rd_Anti-UAV_train_val\track1_test'

# 列出目录下的所有文件夹
subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# 将文件夹名称写入到txt文件中
with open(os.path.join(directory, 'list.txt'), 'w') as f:
    for subdir in subdirectories:
        f.write(subdir + '\n')

print("文件夹名称已写入到 list.txt 文件中。")

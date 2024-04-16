import os

# 指定目录
directory = r'E:\goole_download\3rd_Anti-UAV_train_val\train'

# 列出目录下的所有文件夹名字
subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

# 将文件夹名字写入txt文件
output_file = "subdirs_list.txt"
with open(output_file, "w") as f:
    for subdir in subdirs:
        f.write("'"+subdir+"'," + "\n")

print("Subdirectories list written to", output_file)

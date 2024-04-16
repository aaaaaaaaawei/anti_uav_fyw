import os
# 遍历目录下所有文件夹  删除掉所有的groundtruth.txt
base_dir = 'E:\\goole_download\\3rd_Anti-UAV_train_val\\train'
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    # 如果是文件夹
    if os.path.isdir(folder_path):
        txt_file_path = os.path.join(folder_path, 'groundtruth_rect.txt')
        # 检查文件是否存在，然后删除
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)
            print(f"Deleted {txt_file_path}")
        else:
            print(f"{txt_file_path} does not exist.")
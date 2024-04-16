import json
import os

def process_json_to_txt(json_file_path, output_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract the 'gt_rect' list from the loaded JSON data
    gt_rect = data['gt_rect']

    # Format the 'gt_rect' list to a string that matches the format of 'groundtruth_rect.txt'
    # If a rectangle is empty, output "0,0,0,0", else output as "x,y,width,height"
    formatted_gt_rect = '\n'.join(['0,0,0,0' if not rect else ','.join(map(str, rect)) for rect in gt_rect])

    # Write the formatted string to the output text file
    with open(output_file_path, 'w') as output_file:
        output_file.write(formatted_gt_rect)

base_dir = 'E:\\goole_download\\3rd_Anti-UAV_train_val\\train'
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    # 如果是文件夹
    if os.path.isdir(folder_path):
        # 遍历文件夹下所有文件
        for file_name in os.listdir(folder_path):
            # 如果文件是 JSON 文件
            if file_name.endswith('.json'):
                json_file_path = os.path.join(folder_path, file_name)
                txt_file_path = os.path.join(folder_path, 'groundtruth_rect1.txt')
                #txt_file_path = os.path.splitext(json_file_path)[0] + '.txt'
                # 处理 JSON 文件并写入文本文件
                process_json_to_txt(json_file_path, txt_file_path)



#将train
# 下的json文件中的gt_rect转换成了txt文件
import os
import json

# 定义函数来处理 JSON 文件并写入文本文件
def process_json_to_txt(json_file_path, output_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract the 'gt_rect' list from the loaded JSON data
    gt_rect = data['gt_rect']

    # Format the 'gt_rect' list to a string that matches the format of 'groundtruth_rect.txt'
    # Each line in the output string represents a rectangle, formatted as "x,y,width,height"
    formatted_gt_rect = '\n'.join([','.join(map(str, rect)) for rect in gt_rect])

    # Write the formatted string to the output text file
    with open(output_file_path, 'w') as output_file:
        output_file.write(formatted_gt_rect)
    # with open(json_file_path, 'r') as f:
    #     data = json.load(f)
    #     # print(data)
    #     # print('222222222222222222')
    #     gt_rect = data['gt_rect']
    #     print(gt_rect)
    #     with open(txt_file_path, 'w') as f_txt:
    #         for i in range(0, 4):
    #             f_txt.write('\n'.join([','.join(map(str, sublist)) for sublist in gt_rect]))
    #             if i != 3:
    #                 f_txt.write(',')
    #             if i == 3:
    #                 f_txt.write('\n')

# 遍历目录下所有文件夹
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
                txt_file_path = os.path.join(folder_path, 'groundtruth_rect.txt')
                #txt_file_path = os.path.splitext(json_file_path)[0] + '.txt'
                # 处理 JSON 文件并写入文本文件
                process_json_to_txt(json_file_path, txt_file_path)









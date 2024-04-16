import json
json_file_path='E:\goole_download\\3rd_Anti-UAV_train_val\\train\\20190925_111757_1_7\\IR_label.json'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Extract the 'gt_rect' list from the loaded JSON data
gt_rect = data['gt_rect']

print(gt_rect)
print(len(gt_rect))
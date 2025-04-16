import json
import os
import shutil


bbox_class = {
    "人": 0,
}
label_info = {
    "鼻子": 0,
    "右眼": 1,
    "左眼": 2,
    "右耳": 3,
    "左耳": 4,
    "右肩": 5,
    "左肩": 6,
    "右肘": 7,
    "左肘": 8,
    "右腕": 9,
    "左腕": 10,
    "右臀": 11,
    "左臀": 12,
    "右膝盖": 13,
    "左膝盖": 14,
    "右脚碗": 15,
    "左脚碗": 16,
}


root = "./datas"
save_root = "./datasets"

save_move_person = os.path.join(save_root, "move_person")
if not os.path.exists(save_move_person):
    os.makedirs(save_move_person)


#
# for name in os.listdir(save_root):
#     if not name.endswith(".json"):
#         continue
#     json_path = os.path.join(save_root, name)
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     img_path = json_path.replace(".json", ".jpg")
#     if not os.path.exists(img_path):
#         print("img_path ", img_path, json_path)
#         continue
#     shapes = data["shapes"]
#     img_width = data['imageWidth']  # 图像宽度
#     img_height = data['imageHeight']  # 图像高度
#     person_number = len([shape for shape in shapes if shape["label"] in bbox_class])
#     if person_number != 1:  # 多人则存到对应文件，单独处理
#         shutil.move(img_path, save_move_person)
#         shutil.move(json_path, save_move_person)
#         continue
#     yolo_str = ''
#     # 关键点数据
#     yolo_points = {}
#     for point in label_info:
#         yolo_points[point] = [0.0, 0.0, 0]  # x, y, 可见性（0：不可见，1:可见遮挡, 2:可见不遮挡）
#     for shape in shapes:
#         if shape["label"] in bbox_class:
#             ## 框的信息
#             # 框的类别 ID
#             bbox_class_id, bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm = get_box(shape, img_width, img_height)
#             yolo_str += '{} {:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_class_id, bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm)
#         elif shape["label"] in yolo_points:
#             x, y = shape['points'][0][0], shape['points'][0][1]
#             yolo_points[shape["label"]] = [x / img_width, y / img_height, 2]
#     for (x, y, z) in yolo_points.values():
#         yolo_str += '{:.5f} {:.5f} {} '.format(x, y, z)
#     # 生成 YOLO 格式的 txt 文件
#     suffix = name.split('.')[-2]
#     yolo_txt_path = os.path.join(save_root, suffix + '.txt')
#     with open(yolo_txt_path, 'w', encoding='utf-8') as f:
#         f.write(yolo_str + '\n')
#         f.close()
#     # shutil.move(img_path, save_root)
#     # shutil.move(json_path, save_root)



for name in os.listdir(save_move_person):
    json_path = os.path.join(save_move_person, name)
    if not name.endswith(".json"):
        continue
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    shapes = data["shapes"]
    img_width = data['imageWidth']  # 图像宽度
    img_height = data['imageHeight']  # 图像高度
    yolo_points = {}
    print("json_path", json_path)
    for point in label_info:
        yolo_points[point] = {}  # x, y, 可见性（0：不可见，1:可见遮挡, 2:可见不遮挡）
    idx = 0
    for shape in shapes:
        if shape["label"] in bbox_class:
            bbox_class_id, bbox_center_x_norm, bbox_center_y_norm, bbox_width_norm, bbox_height_norm = get_box(shape, img_width, img_height)
            print(f"[{bbox_class_id},{bbox_center_x_norm},{bbox_center_y_norm},{bbox_width_norm},{bbox_height_norm}],")
        elif shape["label"] in yolo_points:
            if shape["label"] == "鼻子":
                idx += 1
            x, y = shape['points'][0][0], shape['points'][0][1]
            yolo_points[shape["label"]][idx] = [x / img_width, y / img_height, 2]
    for k, v in yolo_points.items():
        print(k, v)
        pass

    suffix = name.split('.')[-2]
    yolo_txt_path = os.path.join(save_move_person, suffix + '.txt')
    test1 = [
        [0, 0.24739583333333334, 0.5514563106796116, 0.3385416666666667, 0.8184466019417476],
        [0, 0.46197916666666666, 0.4854368932038835, 0.22916666666666666, 0.9485436893203884],
    ]
    test2 = [
        [],
        [],
        [],
    ]

    for test in yolo_points.values():
        for i in range(3):
            lst = test.get(i+1, [0, 0, 0])
            test2[i].append(lst)
    yolo_str = ""
    for idx in range(len(test1)):
        yolo_str += '{} {:.5f} {:.5f} {:.5f} {:.5f} '.format(*test1[idx])
        for (x, y, z) in test2[idx]:
            yolo_str += '{:.5f} {:.5f} {} '.format(x, y, z)
        yolo_str += '\n'
    print(yolo_str)
    # with open(yolo_txt_path, 'w', encoding='utf-8') as f:
    #     f.write(yolo_str + '\n')
    #     f.close()

    break
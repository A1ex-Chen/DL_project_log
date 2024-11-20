import os
import json
import cv2
from PIL import ImageOps, Image

root = "./datasets"

for name in os.listdir(root):
    if not name.endswith(".json"):
        continue
    json_path = os.path.join(root, name)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img_path = json_path.replace(".json", ".jpg")
    if not os.path.exists(img_path):
        print("img_path ", img_path, json_path)
        continue
    shapes = data["shapes"]
    img_width = data['imageWidth']  # 图像宽度
    img_height = data['imageHeight']  # 图像高度
    person_number = len([shape for shape in shapes if shape["label"] == "人"])
    if person_number != 1:  # 多人则存到对应文件，单独处理
        print("name", name)


# root = "./datas"
# for file in os.listdir(root):
#     if file.endswith(".json"):
#         continue
#     file_path = os.path.join(root, file)
#     with open(file_path, "rb") as f:
#         f.seek(-2, 2)
#         if f.read() != b"\xff\xd9":  # JPEG
#             print("file_path", file_path)
#             name = file.split(".")[0]
#             img = Image.open(file_path)
#             img = img.convert("RGB")
#             img.save(f"{os.path.join(root, name)}.jpg", "JPEG", subsampling=0, quality=100)
#             f.close()
#             os.remove(file_path)
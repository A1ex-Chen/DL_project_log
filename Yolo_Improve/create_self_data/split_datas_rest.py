import os
import random
import shutil
from tqdm import tqdm

root = "./datas"
target = "./datasets"
target_images = os.path.join(target, "images")
target_labels = os.path.join(target, "labels")

if not os.path.exists(target):
    os.makedirs(target)
    os.makedirs(target_images)
    os.makedirs(target_labels)

for file_name in os.listdir(root):
    if not file_name.endswith(".txt"):
        continue
    img_name = file_name.replace(".txt", ".jpg")
    txt_path = os.path.join(root, file_name)
    img_path = os.path.join(root, img_name)
    if not os.path.exists(img_path):
        continue
    img_save = os.path.join(target_images, img_name)
    txt_save = os.path.join(target_labels, file_name)
    shutil.copy(img_path, img_save)
    shutil.copy(txt_path, txt_save)


test_frac = 0.2
random.seed(123)

img_paths = os.listdir(target_images)
random.shuffle(img_paths)

val_number = int(len(img_paths) * test_frac)
train_files = img_paths[val_number:]
val_files = img_paths[:val_number]

target_images_train = os.path.join(target_images, "train")
target_labels_train = os.path.join(target_labels, "train")
if not os.path.exists(target_images_train):
    os.makedirs(target_images_train)
    os.makedirs(target_labels_train)

train_lst = []

for idx, name in enumerate(train_files):
    path = os.path.join(target_images, name)
    txt_path = path.replace("images", "labels").replace(".jpg", ".txt")
    shutil.move(path, os.path.join(target_images_train, f"{idx}.jpg"))
    shutil.move(txt_path, os.path.join(target_labels_train, f"{idx}.txt"))
    train_lst.append(os.path.join("./images/train", f"{idx}.jpg"))

target_images_val = os.path.join(target_images, "val")
target_labels_val = os.path.join(target_labels, "val")
if not os.path.exists(target_images_val):
    os.makedirs(target_images_val)
    os.makedirs(target_labels_val)

val_lst = []

for idx, name in enumerate(val_files):
    path = os.path.join(target_images, name)
    txt_path = path.replace("images", "labels").replace(".jpg", ".txt")
    shutil.move(path, os.path.join(target_images_val, f"{idx}.jpg"))
    shutil.move(txt_path, os.path.join(target_labels_val, f"{idx}.txt"))
    val_lst.append(os.path.join("./images/val", f"{idx}.jpg"))

with open(os.path.join(target, "train.txt"), 'w', encoding='utf-8') as file:
    file.writelines(item + '\n' for item in train_lst)
    file.close()
with open(os.path.join(target, "val.txt"), 'w', encoding='utf-8') as file:
    file.writelines(item + '\n' for item in val_lst)
    file.close()
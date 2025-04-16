def _save_yolo_image(self, json_data, json_name, image_dir_path, target_dir):
    img_name = json_name.replace('.json', '.png')
    img_path = os.path.join(image_dir_path, target_dir, img_name)
    if not os.path.exists(img_path):
        img = utils.img_b64_to_arr(json_data['imageData'])
        PIL.Image.fromarray(img).save(img_path)
    return img_path

def convert_one(self, json_name):
    json_path = os.path.join(self._json_dir, json_name)
    json_data = json.load(open(json_path))
    print('Converting %s ...' % json_name)
    img_path = self._save_yolo_image(json_data, json_name, self._json_dir, '')
    yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
    self._save_yolo_label(json_name, self._json_dir, '', yolo_obj_list)

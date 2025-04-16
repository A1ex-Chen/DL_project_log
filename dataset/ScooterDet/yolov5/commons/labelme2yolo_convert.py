def convert(self, val_size):
    json_names = [file_name for file_name in os.listdir(self._json_dir) if 
        os.path.isfile(os.path.join(self._json_dir, file_name)) and
        file_name.endswith('.json')]
    folders = [file_name for file_name in os.listdir(self._json_dir) if os.
        path.isdir(os.path.join(self._json_dir, file_name))]
    train_json_names, val_json_names = self._train_test_split(folders,
        json_names, val_size)
    self._make_train_val_dir()
    for target_dir, json_names in zip(('train/', 'val/'), (train_json_names,
        val_json_names)):
        for json_name in json_names:
            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))
            print('Converting %s for %s ...' % (json_name, target_dir.
                replace('/', '')))
            img_path = self._save_yolo_image(json_data, json_name, self.
                _image_dir_path, target_dir)
            yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
            self._save_yolo_label(json_name, self._label_dir_path,
                target_dir, yolo_obj_list)
    print('Generating dataset.yaml file ...')
    self._save_dataset_yaml()

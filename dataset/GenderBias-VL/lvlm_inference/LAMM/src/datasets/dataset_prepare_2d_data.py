def prepare_2d_data(self):
    vision_path_list, caption_list, task_type_list = [], [], []
    with open(self.data_file_path_2d, 'r') as fr:
        json_data = json.load(fr)
    vision_type = 'image'
    for item in json_data:
        if vision_type not in item:
            continue
        one_vision_name, one_caption = item[vision_type], item['conversations']
        task_type = item['task_type'] if 'task_type' in item else 'normal'
        if not one_vision_name.startswith('/'):
            one_vision_path = os.path.join(self.vision_root_path_2d,
                one_vision_name)
        else:
            one_vision_path = one_vision_name
        vision_path_list.append(one_vision_path)
        caption_list.append(one_caption)
        task_type_list.append(task_type)
    print(
        f'[!] collect {len(vision_path_list)} samples (loop x{self.loop_2d}) for image modality training'
        )
    return dict(vision_path_list=vision_path_list, caption_list=
        caption_list, task_type_list=task_type_list)

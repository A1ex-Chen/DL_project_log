def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list
    ):
    txt_path = os.path.join(label_dir_path, target_dir, json_name.replace(
        '.json', '.txt'))
    with open(txt_path, 'w+') as f:
        for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
            yolo_obj_line = ''
            for i in yolo_obj:
                yolo_obj_line += f'{i} '
            yolo_obj_line = yolo_obj_line[:-1]
            if yolo_obj_idx != len(yolo_obj_list) - 1:
                yolo_obj_line += '\n'
            f.write(yolo_obj_line)

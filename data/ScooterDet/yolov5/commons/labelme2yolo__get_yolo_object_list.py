def _get_yolo_object_list(self, json_data, img_path):
    yolo_obj_list = []
    img_h, img_w, _ = cv2.imread(img_path).shape
    for shape in json_data['shapes']:
        if shape['shape_type'] == 'circle':
            yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
        else:
            yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
        yolo_obj_list.append(yolo_obj)
    return yolo_obj_list

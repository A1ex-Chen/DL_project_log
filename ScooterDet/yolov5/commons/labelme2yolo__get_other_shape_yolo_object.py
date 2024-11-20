def _get_other_shape_yolo_object(self, shape, img_h, img_w):
    label_id = self._label_id_map[shape['label']] - 1
    if self._to_seg:
        retval = [label_id]
        for i in shape['points']:
            i[0] = round(float(i[0]) / img_w, 6)
            i[1] = round(float(i[1]) / img_h, 6)
            retval.extend(i)
        return retval

    def __get_object_desc(obj_port_list):
        __get_dist = lambda int_list: max(int_list) - min(int_list)
        x_lists = [port[0] for port in obj_port_list]
        y_lists = [port[1] for port in obj_port_list]
        return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(
            y_lists)
    obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
    yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
    yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
    yolo_w = round(float(obj_w / img_w), 6)
    yolo_h = round(float(obj_h / img_h), 6)
    return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

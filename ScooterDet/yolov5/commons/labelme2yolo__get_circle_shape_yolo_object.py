def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
    label_id = self._label_id_map[shape['label']] - 1
    print(label_id)
    obj_center_x, obj_center_y = shape['points'][0]
    radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 + (
        obj_center_y - shape['points'][1][1]) ** 2)
    if self._to_seg:
        retval = [label_id]
        n_part = radius / 10
        n_part = int(n_part) if n_part > 4 else 4
        n_part2 = n_part << 1
        pt_quad = [None for i in range(0, 4)]
        pt_quad[0] = [[obj_center_x + math.cos(i * math.pi / n_part2) *
            radius, obj_center_y - math.sin(i * math.pi / n_part2) * radius
            ] for i in range(1, n_part)]
        pt_quad[1] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[0]]
        pt_quad[1].reverse()
        pt_quad[3] = [[x1, obj_center_y * 2 - y1] for x1, y1 in pt_quad[0]]
        pt_quad[3].reverse()
        pt_quad[2] = [[obj_center_x * 2 - x1, y1] for x1, y1 in pt_quad[3]]
        pt_quad[2].reverse()
        pt_quad[0].append([obj_center_x, obj_center_y - radius])
        pt_quad[1].append([obj_center_x - radius, obj_center_y])
        pt_quad[2].append([obj_center_x, obj_center_y + radius])
        pt_quad[3].append([obj_center_x + radius, obj_center_y])
        for i in pt_quad:
            for j in i:
                j[0] = round(float(j[0]) / img_w, 6)
                j[1] = round(float(j[1]) / img_h, 6)
                retval.extend(j)
        return retval
    obj_w = 2 * radius
    obj_h = 2 * radius
    yolo_center_x = round(float(obj_center_x / img_w), 6)
    yolo_center_y = round(float(obj_center_y / img_h), 6)
    yolo_w = round(float(obj_w / img_w), 6)
    yolo_h = round(float(obj_h / img_h), 6)
    return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

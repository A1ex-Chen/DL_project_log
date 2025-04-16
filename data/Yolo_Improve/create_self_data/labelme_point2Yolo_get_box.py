def get_box(shape, img_width, img_height):
    bbox_class_id = bbox_class[shape['label']]
    bbox_top_left_x = int(min(shape['points'][0][0], shape['points'][1][0]))
    bbox_bottom_right_x = int(max(shape['points'][0][0], shape['points'][1][0])
        )
    bbox_top_left_y = int(min(shape['points'][0][1], shape['points'][1][1]))
    bbox_bottom_right_y = int(max(shape['points'][0][1], shape['points'][1][1])
        )
    bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
    bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
    bbox_width = bbox_bottom_right_x - bbox_top_left_x
    bbox_height = bbox_bottom_right_y - bbox_top_left_y
    bbox_center_x_norm = bbox_center_x / img_width
    bbox_center_y_norm = bbox_center_y / img_height
    bbox_width_norm = bbox_width / img_width
    bbox_height_norm = bbox_height / img_height
    return (bbox_class_id, bbox_center_x_norm, bbox_center_y_norm,
        bbox_width_norm, bbox_height_norm)

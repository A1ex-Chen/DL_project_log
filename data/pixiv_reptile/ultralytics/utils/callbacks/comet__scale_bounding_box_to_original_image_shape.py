def _scale_bounding_box_to_original_image_shape(box, resized_image_shape,
    original_image_shape, ratio_pad):
    """
    YOLOv8 resizes images during training and the label values are normalized based on this resized shape.

    This function rescales the bounding box labels to the original image shape.
    """
    resized_image_height, resized_image_width = resized_image_shape
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape,
        ratio_pad)
    box = ops.xyxy2xywh(box)
    box[:2] -= box[2:] / 2
    box = box.tolist()
    return box

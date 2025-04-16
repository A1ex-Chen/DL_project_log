def generate_segmentation_from_masks(masks, detected_boxes, image_height,
    image_width, is_image_mask=False):
    """Generates segmentation result from instance masks.

  Args:
    masks: a numpy array of shape [N, mask_height, mask_width] representing the
      instance masks w.r.t. the `detected_boxes`.
    detected_boxes: a numpy array of shape [N, 4] representing the reference
      bounding boxes.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.
    is_image_mask: bool. True: input masks are whole-image masks. False: input
      masks are bounding-box level masks.

  Returns:
    segms: a numpy array of shape [N, image_height, image_width] representing
      the instance masks *pasted* on the image canvas.
  """

    def expand_boxes(boxes, scale):
        """Expands an array of boxes by a given scale."""
        w_half = boxes[:, 2] * 0.5
        h_half = boxes[:, 3] * 0.5
        x_c = boxes[:, 0] + w_half
        y_c = boxes[:, 1] + h_half
        w_half *= scale
        h_half *= scale
        boxes_exp = np.zeros(boxes.shape)
        boxes_exp[:, 0] = x_c - w_half
        boxes_exp[:, 2] = x_c + w_half
        boxes_exp[:, 1] = y_c - h_half
        boxes_exp[:, 3] = y_c + h_half
        return boxes_exp
    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width, (mask_height + 2.0) /
        mask_height)
    ref_boxes = expand_boxes(detected_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []
    for mask_ind, mask in enumerate(masks):
        im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        if is_image_mask:
            im_mask[:, :] = mask[:, :]
        else:
            padded_mask[1:-1, 1:-1] = mask[:, :]
            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)
            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)
            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, image_width)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, image_height)
            im_mask[y_0:y_1, x_0:x_1] = mask[y_0 - ref_box[1]:y_1 - ref_box
                [1], x_0 - ref_box[0]:x_1 - ref_box[0]]
        segms.append(im_mask)
    segms = np.array(segms)
    assert masks.shape[0] == segms.shape[0]
    return segms

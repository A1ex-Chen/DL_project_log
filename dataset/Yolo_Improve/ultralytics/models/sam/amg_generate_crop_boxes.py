def generate_crop_boxes(im_size: Tuple[int, ...], n_layers: int,
    overlap_ratio: float) ->Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes.

    Each layer has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)
    crop_boxes.append([0, 0, im_w, im_h])
    layer_idxs.append(0)

    def crop_len(orig_len, n_crops, overlap):
        """Crops bounding boxes to the size of the input image."""
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)
        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(
            n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(
            n_crops_per_side)]
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)
    return crop_boxes, layer_idxs

def filter_anno_by_mask(image_anno, mask):
    img_filtered_annotations = {}
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][mask]
    return img_filtered_annotations

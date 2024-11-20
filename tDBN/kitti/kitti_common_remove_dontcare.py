def remove_dontcare(image_anno):
    img_filtered_annotations = {}
    relevant_annotation_indices = [i for i, x in enumerate(image_anno[
        'name']) if x != 'DontCare']
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][
            relevant_annotation_indices]
    return img_filtered_annotations

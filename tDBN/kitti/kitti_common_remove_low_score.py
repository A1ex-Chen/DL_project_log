def remove_low_score(image_anno, thresh):
    img_filtered_annotations = {}
    relevant_annotation_indices = [i for i, s in enumerate(image_anno[
        'score']) if s >= thresh]
    for key in image_anno.keys():
        img_filtered_annotations[key] = image_anno[key][
            relevant_annotation_indices]
    return img_filtered_annotations

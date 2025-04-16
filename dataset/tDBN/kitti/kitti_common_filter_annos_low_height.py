def filter_annos_low_height(image_annos, thresh):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [i for i, s in enumerate(anno['bbox']
            ) if s[3] - s[1] >= thresh]
        for key in anno.keys():
            img_filtered_annotations[key] = anno[key][
                relevant_annotation_indices]
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos

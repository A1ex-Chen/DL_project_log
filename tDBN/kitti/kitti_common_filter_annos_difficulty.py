def filter_annos_difficulty(image_annos, used_difficulty):
    new_image_annos = []
    for anno in image_annos:
        img_filtered_annotations = {}
        relevant_annotation_indices = [i for i, x in enumerate(anno[
            'difficulty']) if x in used_difficulty]
        for key in anno.keys():
            img_filtered_annotations[key] = anno[key][
                relevant_annotation_indices]
        new_image_annos.append(img_filtered_annotations)
    return new_image_annos

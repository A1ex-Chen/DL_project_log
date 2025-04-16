def create_coco_format_dataset(images, annotations,
    regenerate_annotation_id=True):
    """Creates COCO format dataset with COCO format images and annotations."""
    if regenerate_annotation_id:
        for i in range(len(annotations)):
            annotations[i]['id'] = i + 1
    categories = _extract_categories(annotations)
    dataset = {'images': images, 'annotations': annotations, 'categories':
        categories}
    return dataset

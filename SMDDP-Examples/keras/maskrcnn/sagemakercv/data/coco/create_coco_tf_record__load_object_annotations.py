def _load_object_annotations(object_annotations_file):
    with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)
    images = obj_annotations['images']
    category_index = label_map_util.create_category_index(obj_annotations[
        'categories'])
    img_to_obj_annotation = collections.defaultdict(list)
    tf.compat.v1.logging.info('Building bounding box index.')
    for annotation in obj_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_obj_annotation[image_id].append(annotation)
    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1
    tf.compat.v1.logging.info('%d images are missing bboxes.',
        missing_annotation_count)
    return images, img_to_obj_annotation, category_index

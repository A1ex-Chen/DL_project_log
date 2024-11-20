def _load_caption_annotations(caption_annotations_file):
    with tf.io.gfile.GFile(caption_annotations_file, 'r') as fid:
        caption_annotations = json.load(fid)
    img_to_caption_annotation = collections.defaultdict(list)
    tf.compat.v1.logging.info('Building caption index.')
    for annotation in caption_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_caption_annotation[image_id].append(annotation)
    missing_annotation_count = 0
    images = caption_annotations['images']
    for image in images:
        image_id = image['id']
        if image_id not in img_to_caption_annotation:
            missing_annotation_count += 1
    tf.compat.v1.logging.info('%d images are missing captions.',
        missing_annotation_count)
    return img_to_caption_annotation

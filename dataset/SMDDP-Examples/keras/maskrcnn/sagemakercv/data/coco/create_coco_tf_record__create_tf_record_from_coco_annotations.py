def _create_tf_record_from_coco_annotations(object_annotations_file,
    caption_annotations_file, image_dir, output_path, include_masks, num_shards
    ):
    """Loads COCO annotation json files and converts to tf.Record format.

  Args:
    object_annotations_file: JSON file containing bounding box annotations.
    caption_annotations_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    include_masks: Whether to include instance segmentations masks
      (PNG encoded) in the result. default: False.
    num_shards: Number of output files to create.
  """
    tf.compat.v1.logging.info('writing to output path: %s', output_path)
    writers = [tf.io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
        (i, num_shards)) for i in range(num_shards)]
    images, img_to_obj_annotation, category_index = _load_object_annotations(
        object_annotations_file)
    img_to_caption_annotation = _load_caption_annotations(
        caption_annotations_file)
    pool = multiprocessing.Pool()
    total_num_annotations_skipped = 0
    for idx, (_, tf_example, num_annotations_skipped) in enumerate(pool.
        imap(_pool_create_tf_example, [(image, img_to_obj_annotation[image[
        'id']], img_to_caption_annotation[image['id']], image_dir,
        category_index, include_masks) for image in images])):
        if idx % 1000 == 0:
            tf.compat.v1.logging.info('On image %d of %d', idx, len(images))
        total_num_annotations_skipped += num_annotations_skipped
        writers[idx % num_shards].write(tf_example.SerializeToString())
    pool.close()
    pool.join()
    for writer in writers:
        writer.close()
    tf.compat.v1.logging.info('Finished writing, skipped %d annotations.',
        total_num_annotations_skipped)

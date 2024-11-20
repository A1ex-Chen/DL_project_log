def main(_):
    assert FLAGS.train_image_dir, '`train_image_dir` missing.'
    assert FLAGS.val_image_dir, '`val_image_dir` missing.'
    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'train')
    val_output_path = os.path.join(FLAGS.output_dir, 'val')
    testdev_output_path = os.path.join(FLAGS.output_dir, 'test-dev')
    _create_tf_record_from_coco_annotations(FLAGS.
        train_object_annotations_file, FLAGS.train_caption_annotations_file,
        FLAGS.train_image_dir, train_output_path, FLAGS.include_masks,
        num_shards=512)
    _create_tf_record_from_coco_annotations(FLAGS.
        val_object_annotations_file, FLAGS.val_caption_annotations_file,
        FLAGS.val_image_dir, val_output_path, FLAGS.include_masks,
        num_shards=512)

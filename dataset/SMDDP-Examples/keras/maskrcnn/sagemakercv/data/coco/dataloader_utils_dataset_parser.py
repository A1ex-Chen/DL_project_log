def dataset_parser(value, mode, params, use_instance_mask, seed=None,
    regenerate_source_id=False):
    """Parse data to a fixed dimension input image and learning targets.

    Args:
    value: A dictionary contains an image and groundtruth annotations.

    Returns:
    features: a dictionary that contains the image and auxiliary
      information. The following describes {key: value} pairs in the
      dictionary.
      image: Image tensor that is preproessed to have normalized value and
        fixed dimension [image_size, image_size, 3]
      image_info: image information that includes the original height and
        width, the scale of the proccessed image to the original image, and
        the scaled height and width.
      source_ids: Source image id. Default value -1 if the source id is
        empty in the groundtruth annotation.
    labels: a dictionary that contains auxiliary information plus (optional)
      labels. The following describes {key: value} pairs in the dictionary.
      `labels` is only for training.
      score_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors]. The height_l and width_l
        represent the dimension of objectiveness score at l-th level.
      box_targets_dict: ordered dictionary with keys
        [min_level, min_level+1, ..., max_level]. The values are tensor with
        shape [height_l, width_l, num_anchors * 4]. The height_l and
        width_l represent the dimension of bounding box regression output at
        l-th level.
      gt_boxes: Groundtruth bounding box annotations. The box is represented
         in [y1, x1, y2, x2] format. The tennsor is padded with -1 to the
         fixed dimension [MAX_NUM_INSTANCES, 4].
      gt_classes: Groundtruth classes annotations. The tennsor is padded
        with -1 to the fixed dimension [MAX_NUM_INSTANCES].
      cropped_gt_masks: groundtrugh masks cropped by the bounding box and
        resized to a fixed size determined by params['gt_mask_size']
      regenerate_source_id: `bool`, if True TFExampleParser will use hashed
        value of `image/encoded` for `image/source_id`.
    """
    if mode not in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.
        PREDICT, tf.estimator.ModeKeys.EVAL]:
        raise ValueError('Unknown execution mode received: %s' % mode)

    def create_example_decoder():
        return TfExampleDecoder(use_instance_mask=use_instance_mask,
            regenerate_source_id=regenerate_source_id)
    example_decoder = create_example_decoder()
    with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        data['groundtruth_is_crowd'] = process_groundtruth_is_crowd(data)
        image = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)
        source_id = process_source_id(data['source_id'])
        if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
            features = {'source_ids': source_id}
            if params['visualize_images_summary']:
                features['orig_images'] = tf.image.resize(image, params[
                    'image_size'])
            features['images'], features['image_info'
                ], _, _ = preprocess_image(image, boxes=None,
                instance_masks=None, image_size=params['image_size'],
                max_level=params['max_level'], augment_input_data=False,
                seed=seed)
            if params['include_groundtruth_in_features']:
                labels = prepare_labels_for_eval(data, target_num_instances
                    =MAX_NUM_INSTANCES, target_polygon_list_len=
                    MAX_NUM_POLYGON_LIST_LEN, use_instance_mask=
                    use_instance_mask)
                return {'features': features, 'labels': labels}
            else:
                return {'features': features}
        elif mode == tf.estimator.ModeKeys.TRAIN:
            labels = {}
            features = {'source_ids': source_id}
            boxes, classes, indices, instance_masks = (
                process_boxes_classes_indices_for_training(data,
                skip_crowd_during_training=params[
                'skip_crowd_during_training'], use_category=params[
                'use_category'], use_instance_mask=use_instance_mask))
            orig_image_size = tf.shape(image)[:2]
            image, image_info, boxes, instance_masks = preprocess_image(image,
                boxes=boxes, instance_masks=instance_masks, image_size=
                params['image_size'], max_level=params['max_level'],
                augment_input_data=params['augment_input_data'], seed=seed)
            features.update({'images': image, 'image_info': image_info})
            padded_image_size = image.get_shape().as_list()[:2]
            if use_instance_mask:
                instance_masks = tf.expand_dims(instance_masks, -1)
                orig_boxes = boxes * image_info[2]
                labels['cropped_gt_masks'] = process_gt_masks_for_training(
                    instance_masks, orig_boxes, gt_mask_size=params[
                    'gt_mask_size'], padded_image_size=orig_image_size,
                    max_num_instances=MAX_NUM_INSTANCES)
            (score_targets, box_targets
                ), input_anchor = process_targets_for_training(
                padded_image_size=padded_image_size, boxes=boxes, classes=
                classes, params=params)
            additional_labels = process_labels_for_training(image_info,
                boxes, classes, score_targets, box_targets,
                max_num_instances=MAX_NUM_INSTANCES, min_level=params[
                'min_level'], max_level=params['max_level'])
            labels.update(additional_labels)
            FAKE_FEATURES = False
            if FAKE_FEATURES:
                labels['source_ids'] = tf.ones(shape=(), dtype=tf.float32)
                labels['images'] = tf.ones(shape=(1024, 1024, 3), dtype=tf.
                    float32)
                labels['image_info'] = tf.ones(shape=(5,), dtype=tf.float32)
            FAKE_LABELS = False
            if FAKE_LABELS:
                labels['cropped_gt_masks'] = tf.ones(shape=(100, 116, 116),
                    dtype=tf.float32)
                labels['gt_boxes'] = tf.ones(shape=(100, 4), dtype=tf.float32)
                labels['gt_classes'] = tf.ones(shape=(100, 1), dtype=tf.float32
                    )
                idx = 1
                for dim in [256, 128, 64, 32, 16]:
                    idx += 1
                    labels['score_targets_%d' % idx] = tf.ones(shape=(dim,
                        dim, 3), dtype=tf.float32)
                    labels['box_targets_%d' % idx] = tf.ones(shape=(dim,
                        dim, 12), dtype=tf.float32)
            return features, labels

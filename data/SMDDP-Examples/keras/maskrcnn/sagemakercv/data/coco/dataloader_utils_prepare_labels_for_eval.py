def prepare_labels_for_eval(data, target_num_instances=MAX_NUM_INSTANCES,
    target_polygon_list_len=MAX_NUM_POLYGON_LIST_LEN, use_instance_mask=False):
    """Create labels dict for infeed from data of tf.Example."""
    image = data['image']
    height = tf.shape(input=image)[0]
    width = tf.shape(input=image)[1]
    boxes = data['groundtruth_boxes']
    classes = tf.cast(data['groundtruth_classes'], dtype=tf.float32)
    num_labels = tf.shape(input=classes)[0]
    boxes = preprocess_ops.pad_to_fixed_size(boxes, -1, [
        target_num_instances, 4])
    classes = preprocess_ops.pad_to_fixed_size(classes, -1, [
        target_num_instances, 1])
    is_crowd = tf.cast(data['groundtruth_is_crowd'], dtype=tf.float32)
    is_crowd = preprocess_ops.pad_to_fixed_size(is_crowd, 0, [
        target_num_instances, 1])
    labels = dict()
    labels['width'] = width
    labels['height'] = height
    labels['groundtruth_boxes'] = boxes
    labels['groundtruth_classes'] = classes
    labels['num_groundtruth_labels'] = num_labels
    labels['groundtruth_is_crowd'] = is_crowd
    """if use_instance_mask:
        data['groundtruth_masks'] = preprocess_ops.pad_to_fixed_size(
            data=data['groundtruth_polygons'],
            pad_value=POLYGON_PAD_VALUE,
            output_shape=[target_polygon_list_len, 1]
        )"""
    return labels

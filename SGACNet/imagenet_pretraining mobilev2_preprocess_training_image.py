def preprocess_training_image(image, label):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, bbox = tf.image.sample_distorted_bounding_box(
        image_size=tf.shape(image), bounding_boxes=bbox, min_object_covered
        =0.1, aspect_ratio_range=[3.0 / 4, 4.0 / 3], area_range=[0.08, 1.0],
        max_attempts=1, use_image_if_no_bounding_boxes=True)
    image = tf.slice(image, bbox_begin, bbox_size)
    image = tf.image.resize(image, (224, 224)) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    image = image - mean
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    std = tf.reshape(std, [1, 1, 3])
    image = image / std
    image = tf.image.random_flip_left_right(image)
    image = tf.transpose(image, perm=[2, 0, 1])
    return image, label

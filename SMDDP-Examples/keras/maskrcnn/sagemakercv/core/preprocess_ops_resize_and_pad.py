def resize_and_pad(image, target_size, stride, boxes=None, masks=None):
    """Resize and pad images, boxes and masks.

    Resize and pad images, (optionally boxes and masks) given the desired output
    size of the image and stride size.

    Here are the preprocessing steps.
    1. For a given image, keep its aspect ratio and rescale the image to make it
     the largest rectangle to be bounded by the rectangle specified by the
     `target_size`.
    2. Pad the rescaled image such that the height and width of the image become
     the smallest multiple of the stride that is larger or equal to the desired
     output diemension.

    Args:
    image: an image tensor of shape [original_height, original_width, 3].
    target_size: a tuple of two integers indicating the desired output
      image size. Note that the actual output size could be different from this.
    stride: the stride of the backbone network. Each of the output image sides
      must be the multiple of this.
    boxes: (Optional) a tensor of shape [num_boxes, 4] represneting the box
      corners in normalized coordinates.
    masks: (Optional) a tensor of shape [num_masks, height, width]
      representing the object masks. Note that the size of the mask is the
      same as the image.

    Returns:
    image: the processed image tensor after being resized and padded.
    image_info: a tensor of shape [5] which encodes the height, width before
      and after resizing and the scaling factor.
    boxes: None or the processed box tensor after being resized and padded.
      After the processing, boxes will be in the absolute coordinates w.r.t.
      the scaled image.
    masks: None or the processed mask tensor after being resized and padded.
    """
    input_height, input_width, _ = tf.unstack(tf.cast(tf.shape(input=image),
        dtype=tf.float32), axis=0)
    target_height, target_width = target_size
    scale_if_resize_height = target_height / input_height
    scale_if_resize_width = target_width / input_width
    scale = tf.minimum(scale_if_resize_height, scale_if_resize_width)
    scaled_height = tf.cast(scale * input_height, dtype=tf.int32)
    scaled_width = tf.cast(scale * input_width, dtype=tf.int32)
    image = tf.image.resize(image, [scaled_height, scaled_width], method=tf
        .image.ResizeMethod.BILINEAR)
    padded_height = int(math.ceil(target_height * 1.0 / stride) * stride)
    padded_width = int(math.ceil(target_width * 1.0 / stride) * stride)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_height,
        padded_width)
    image.set_shape([padded_height, padded_width, 3])
    image_info = tf.stack([tf.cast(scaled_height, dtype=tf.float32), tf.
        cast(scaled_width, dtype=tf.float32), 1.0 / scale, input_height,
        input_width])
    if boxes is not None:
        normalized_box_list = preprocessor.box_list.BoxList(boxes)
        scaled_boxes = preprocessor.box_list_scale(normalized_box_list,
            scaled_height, scaled_width).get()
    else:
        scaled_boxes = None
    if masks is not None:
        scaled_masks = tf.image.resize(tf.expand_dims(masks, -1), [
            scaled_height, scaled_width], method=tf.image.ResizeMethod.
            NEAREST_NEIGHBOR)
        num_masks = tf.shape(input=scaled_masks)[0]
        scaled_masks = tf.cond(pred=tf.greater(num_masks, 0), true_fn=lambda :
            tf.image.pad_to_bounding_box(scaled_masks, 0, 0, padded_height,
            padded_width), false_fn=lambda : tf.zeros([0, padded_height,
            padded_width, 1]))
    else:
        scaled_masks = None
    return image, image_info, scaled_boxes, scaled_masks

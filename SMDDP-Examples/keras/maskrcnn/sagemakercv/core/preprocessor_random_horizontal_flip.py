def random_horizontal_flip(image, boxes=None, masks=None, keypoints=None,
    keypoint_flip_permutation=None, seed=None):
    """Randomly flips the image and detections horizontally.

  The probability of flipping the image is 50%.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: (optional) rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks. The masks
           are of the same height, width as the input `image`.
    keypoints: (optional) rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]. The keypoints are in y-x
               normalized coordinates.
    keypoint_flip_permutation: rank 1 int32 tensor containing the keypoint flip
                               permutation.
    seed: random seed

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, and keypoint_flip_permutation are not None,
    the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]

  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
  """

    def _flip_image(image):
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped
    if keypoints is not None and keypoint_flip_permutation is None:
        raise ValueError(
            'keypoints are provided but keypoints_flip_permutation is not provided'
            )
    result = []
    do_a_flip_random = tf.greater(tf.random.uniform([], seed=seed), 0.5)
    image = tf.cond(pred=do_a_flip_random, true_fn=lambda : _flip_image(
        image), false_fn=lambda : image)
    result.append(image)
    if boxes is not None:
        boxes = tf.cond(pred=do_a_flip_random, true_fn=lambda :
            _flip_boxes_left_right(boxes), false_fn=lambda : boxes)
        result.append(boxes)
    if masks is not None:
        masks = tf.cond(pred=do_a_flip_random, true_fn=lambda :
            _flip_masks_left_right(masks), false_fn=lambda : masks)
        result.append(masks)
    if keypoints is not None and keypoint_flip_permutation is not None:
        permutation = keypoint_flip_permutation
        keypoints = tf.cond(pred=do_a_flip_random, true_fn=lambda :
            keypoint_flip_horizontal(keypoints, 0.5, permutation), false_fn
            =lambda : keypoints)
        result.append(keypoints)
    return tuple(result)

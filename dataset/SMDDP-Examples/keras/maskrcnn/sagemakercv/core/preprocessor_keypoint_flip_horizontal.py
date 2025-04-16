def keypoint_flip_horizontal(keypoints, flip_point, flip_permutation, scope
    =None):
    """Flips the keypoints horizontally around the flip_point.

  This operation flips the x coordinate for each keypoint around the flip_point
  and also permutes the keypoints in a manner specified by flip_permutation.

  Args:
    keypoints: a tensor of shape [num_instances, num_keypoints, 2]
    flip_point:  (float) scalar tensor representing the x coordinate to flip the
      keypoints around.
    flip_permutation: rank 1 int32 tensor containing the keypoint flip
      permutation. This specifies the mapping from original keypoint indices
      to the flipped keypoint indices. This is used primarily for keypoints
      that are not reflection invariant. E.g. Suppose there are 3 keypoints
      representing ['head', 'right_eye', 'left_eye'], then a logical choice for
      flip_permutation might be [0, 2, 1] since we want to swap the 'left_eye'
      and 'right_eye' after a horizontal flip.
    scope: name scope.

  Returns:
    new_keypoints: a tensor of shape [num_instances, num_keypoints, 2]
  """
    keypoints = tf.transpose(a=keypoints, perm=[1, 0, 2])
    keypoints = tf.gather(keypoints, flip_permutation)
    v, u = tf.split(value=keypoints, num_or_size_splits=2, axis=2)
    u = flip_point * 2.0 - u
    new_keypoints = tf.concat([v, u], 2)
    new_keypoints = tf.transpose(a=new_keypoints, perm=[1, 0, 2])
    return new_keypoints

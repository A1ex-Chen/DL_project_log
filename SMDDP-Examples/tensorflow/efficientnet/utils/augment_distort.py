def distort(self, image: tf.Tensor) ->tf.Tensor:
    """Applies the RandAugment policy to `image`.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      The augmented version of `image`.
    """
    input_image_type = image.dtype
    if input_image_type != tf.uint8:
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.cast(image, dtype=tf.uint8)
    replace_value = [128] * 3
    min_prob, max_prob = 0.2, 0.8
    for _ in range(self.num_layers):
        op_to_select = tf.random.uniform([], maxval=len(self.available_ops) +
            1, dtype=tf.int32)
        branch_fns = []
        for i, op_name in enumerate(self.available_ops):
            prob = tf.random.uniform([], minval=min_prob, maxval=max_prob,
                dtype=tf.float32)
            func, _, args = _parse_policy_info(op_name, prob, self.
                magnitude, replace_value, self.cutout_const, self.
                translate_const)
            branch_fns.append((i, lambda selected_func=func, selected_args=
                args: selected_func(image, *selected_args)))
        image = tf.switch_case(branch_index=op_to_select, branch_fns=
            branch_fns, default=lambda : tf.identity(image))
    image = tf.cast(image, dtype=input_image_type)
    return image

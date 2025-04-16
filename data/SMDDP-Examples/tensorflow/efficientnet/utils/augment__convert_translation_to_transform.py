def _convert_translation_to_transform(translations: tf.Tensor) ->tf.Tensor:
    """Converts translations to a projective transform.

  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]

  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.

  Returns:
    The transformation matrix of shape (num_images, 8).

  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.

  """
    translations = tf.convert_to_tensor(translations, dtype=tf.float32)
    if translations.get_shape().ndims is None:
        raise TypeError('translations rank must be statically known')
    elif len(translations.get_shape()) == 1:
        translations = translations[None]
    elif len(translations.get_shape()) != 2:
        raise TypeError('translations should have rank 1 or 2.')
    num_translations = tf.shape(translations)[0]
    return tf.concat(values=[tf.ones((num_translations, 1), tf.dtypes.
        float32), tf.zeros((num_translations, 1), tf.dtypes.float32), -
        translations[:, 0, None], tf.zeros((num_translations, 1), tf.dtypes
        .float32), tf.ones((num_translations, 1), tf.dtypes.float32), -
        translations[:, 1, None], tf.zeros((num_translations, 2), tf.dtypes
        .float32)], axis=1)

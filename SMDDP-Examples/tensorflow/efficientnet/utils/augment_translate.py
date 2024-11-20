def translate(image: tf.Tensor, translations) ->tf.Tensor:
    """Translates image(s) by provided vectors.

  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].

  Returns:
    The translated version of the image.

  """
    transforms = _convert_translation_to_transform(translations)
    return transform(image, transforms=transforms)

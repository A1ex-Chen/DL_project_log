# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoAugment and RandAugment policies for enhanced image preprocessing.

AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math
import tensorflow as tf
from typing import Any, Dict, List, Optional, Text, Tuple

from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.








































































NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})





  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
  """Implements Equalize function from PIL using TF ops."""

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], axis=2)
  return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level: float):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level: float):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level: float):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _mult_to_arg(level: float, multiplier: float = 1.):
  return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func: Any,
                          image: tf.Tensor,
                          args: Any,
                          prob: float):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(
      should_apply_op,
      lambda: func(image, *args),
      lambda: image)
  return augmented_image


def select_and_apply_random_policy(policies: Any, image: tf.Tensor):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})


def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
  }
  return args


def _parse_policy_info(name: Text,
                       prob: float,
                       level: float,
                       replace_value: List[int],
                       cutout_const: float,
                       translate_const: float) -> Tuple[Any, float, Any]:
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""



class AutoAugment(ImageAugment):
  """Applies the AutoAugment policy to images.

    AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
  """



  @staticmethod

  @staticmethod

  @staticmethod


class RandAugment(ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """



    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image: tf.Tensor, factor: float) -> tf.Tensor:
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image: tf.Tensor) -> tf.Tensor:
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255


    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def invert(image: tf.Tensor) -> tf.Tensor:
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def wrap(image: tf.Tensor) -> tf.Tensor:
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], axis=2)
  return extended


def unwrap(image: tf.Tensor, replace: int) -> tf.Tensor:
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level: float):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level: float):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level: float):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level: float):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _translate_level_to_arg(level: float, translate_const: float):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)


def _mult_to_arg(level: float, multiplier: float = 1.):
  return (int((level / _MAX_LEVEL) * multiplier),)


def _apply_func_with_prob(func: Any,
                          image: tf.Tensor,
                          args: Any,
                          prob: float):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(
      should_apply_op,
      lambda: func(image, *args),
      lambda: image)
  return augmented_image


def select_and_apply_random_policy(policies: Any, image: tf.Tensor):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Cutout': cutout,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
    'Cutout',
})


def level_to_arg(cutout_const: float, translate_const: float):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  solarize_add_arg = lambda level: _mult_to_arg(level, 110)
  cutout_arg = lambda level: _mult_to_arg(level, cutout_const)
  translate_arg = lambda level: _translate_level_to_arg(level, translate_const)

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Invert': no_arg,
      'Rotate': _rotate_level_to_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'SolarizeAdd': solarize_add_arg,
      'Color': _enhance_level_to_arg,
      'Contrast': _enhance_level_to_arg,
      'Brightness': _enhance_level_to_arg,
      'Sharpness': _enhance_level_to_arg,
      'ShearX': _shear_level_to_arg,
      'ShearY': _shear_level_to_arg,
      'Cutout': cutout_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
  }
  return args


def _parse_policy_info(name: Text,
                       prob: float,
                       level: float,
                       replace_value: List[int],
                       cutout_const: float,
                       translate_const: float) -> Tuple[Any, float, Any]:
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg(cutout_const, translate_const)[name](level)

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  return func, prob, args


class ImageAugment(object):
  """Image augmentation class for applying image distortions."""

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Given an image tensor, returns a distorted image with the same shape.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      The augmented version of `image`.
    """
    raise NotImplementedError()


class AutoAugment(ImageAugment):
  """Applies the AutoAugment policy to images.

    AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.
  """

  def __init__(self,
               augmentation_name: Text = 'v0',
               policies: Optional[Dict[Text, Any]] = None,
               cutout_const: float = 100,
               translate_const: float = 250):
    """Applies the AutoAugment policy to images.

    Args:
      augmentation_name: The name of the AutoAugment policy to use. The
        available options are `v0` and `test`. `v0` is the policy used for all
        of the results in the paper and was found to achieve the best results on
        the COCO dataset. `v1`, `v2` and `v3` are additional good policies found
        on the COCO dataset that have slight variation in what operations were
        used during the search procedure along with how many operations are
        applied in parallel to a single image (2 vs 3).
      policies: list of lists of tuples in the form `(func, prob, level)`,
        `func` is a string name of the augmentation function, `prob` is the
        probability of applying the `func` operation, `level` is the input
        argument for `func`.
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
    """
    super(AutoAugment, self).__init__()


    if policies is None:
      self.available_policies = {
          'v0': self.policy_v0(),
          'test': self.policy_test(),
          'simple': self.policy_simple(),
      }

    if augmentation_name not in self.available_policies:
      raise ValueError(
          'Invalid augmentation_name: {}'.format(augmentation_name))

    self.augmentation_name = augmentation_name
    self.policies = self.available_policies[augmentation_name]
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)

  def distort(self, image: tf.Tensor) -> tf.Tensor:
    """Applies the AutoAugment policy to `image`.

    AutoAugment is from the paper: https://arxiv.org/abs/1805.09501.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.

    Returns:
      A version of image that now has data augmentation applied to it based on
      the `policies` pass into the function.
    """
    input_image_type = image.dtype

    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3

    # func is the string name of the augmentation function, prob is the
    # probability of applying the operation and level is the parameter
    # associated with the tf op.

    # tf_policies are functions that take in an image and return an augmented
    # image.
    tf_policies = []
    for policy in self.policies:
      tf_policy = []
      # Link string name to the correct python function and make sure the
      # correct argument is passed into that function.
      for policy_info in policy:
        policy_info = list(policy_info) + [
            replace_value, self.cutout_const, self.translate_const
        ]
        tf_policy.append(_parse_policy_info(*policy_info))
      # Now build the tf policy that will apply the augmentation procedue
      # on image.

        return final_policy

      tf_policies.append(make_final_policy(tf_policy))

    image = select_and_apply_random_policy(tf_policies, image)
    image = tf.cast(image, dtype=input_image_type)
    return image

  @staticmethod
  def policy_v0():
    """Autoaugment policy that was used in AutoAugment Paper.

    Each tuple is an augmentation operation of the form
    (operation, probability, magnitude). Each element in policy is a
    sub-policy that will be applied sequentially on the image.

    Returns:
      the policy.
    """

    # TODO(dankondratyuk): tensorflow_addons defines custom ops, which
    # for some reason are not included when building/linking
    # This results in the error, "Op type not registered
    # 'Addons>ImageProjectiveTransformV2' in binary" when running on borg TPUs
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
        [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
        [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
        [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
        [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
        [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
        [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
    ]
    return policy

  @staticmethod
  def policy_simple():
    """Same as `policy_v0`, except with custom ops removed."""

    policy = [
        [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
        [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
        [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
        [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
        [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
        [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
        [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
        [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
        [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
        [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
        [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
        [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
        [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
    ]
    return policy

  @staticmethod
  def policy_test():
    """Autoaugment test policy for debugging."""
    policy = [
        [('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)],
    ]
    return policy


class RandAugment(ImageAugment):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """

  def __init__(self,
               num_layers: int = 2,
               magnitude: float = 10.,
               cutout_const: float = 40.,
               translate_const: float = 100.):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      cutout_const: multiplier for applying cutout.
      translate_const: multiplier for applying translation.
    """
    super(RandAugment, self).__init__()

    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.cutout_const = float(cutout_const)
    self.translate_const = float(translate_const)
    self.available_ops = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
        'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
        'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
    ]

  def distort(self, image: tf.Tensor) -> tf.Tensor:
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
      op_to_select = tf.random.uniform(
          [], maxval=len(self.available_ops) + 1, dtype=tf.int32)

      branch_fns = []
      for (i, op_name) in enumerate(self.available_ops):
        prob = tf.random.uniform([],
                                 minval=min_prob,
                                 maxval=max_prob,
                                 dtype=tf.float32)
        func, _, args = _parse_policy_info(op_name,
                                           prob,
                                           self.magnitude,
                                           replace_value,
                                           self.cutout_const,
                                           self.translate_const)
        branch_fns.append((
            i,
            # pylint:disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args)))
        # pylint:enable=g-long-lambda

      image = tf.switch_case(branch_index=op_to_select,
                             branch_fns=branch_fns,
                             default=lambda: tf.identity(image))

    image = tf.cast(image, dtype=input_image_type)
    return image
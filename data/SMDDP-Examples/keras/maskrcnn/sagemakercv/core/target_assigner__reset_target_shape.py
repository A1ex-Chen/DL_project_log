def _reset_target_shape(self, target, num_anchors):
    """Sets the static shape of the target.

        Args:
          target: the target tensor. Its first dimension will be overwritten.
          num_anchors: the number of anchors, which is used to override the target's
            first dimension.

        Returns:
          A tensor with the shape info filled in.
        """
    target_shape = target.get_shape().as_list()
    target_shape[0] = num_anchors
    target.set_shape(target_shape)
    return target

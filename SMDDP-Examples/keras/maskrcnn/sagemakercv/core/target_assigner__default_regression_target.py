def _default_regression_target(self):
    """Returns the default target for anchors to regress to.

        Default regression targets are set to zero (though in
        this implementation what these targets are set to should
        not matter as the regression weight of any box set to
        regress to the default target is zero).

        Returns:
          default_target: a float32 tensor with shape [1, box_code_dimension]
        """
    return tf.constant([self._box_coder.code_size * [0]], tf.float32)

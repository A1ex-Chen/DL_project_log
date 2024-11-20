def call(self, inputs, **kwargs):
    """
        Returns:
        class_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes], representing the class predictions.
        box_outputs: a tensor with a shape of
          [batch_size, num_rois, num_classes * 4], representing the box predictions.
        box_features: a tensor with a shape of
          [batch_size, num_rois, mlp_head_dim], representing the box features.
        """
    batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()
    net = tf.reshape(inputs, [batch_size, num_rois, height * width * filters])
    net = self._bbox_dense_0(net)
    box_features = self._bbox_dense_1(net)
    class_outputs = self._dense_class(box_features)
    box_outputs = self._dense_box(box_features)
    return class_outputs, box_outputs, box_features

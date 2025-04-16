def call(self, inputs, class_indices, **kwargs):
    """
        class_indices: a Tensor of shape [batch_size, num_rois], indicating
          which class the ROI is.
        Returns:
        mask_outputs: a tensor with a shape of
          [batch_size, num_masks, mask_height, mask_width],
          representing the mask predictions.
        fg_gather_indices: a tensor with a shape of [batch_size, num_masks, 2],
          representing the fg mask targets.
        Raises:
        ValueError: If boxes is not a rank-3 tensor or the last dimension of
          boxes is not 4.
        """
    batch_size, num_rois, height, width, filters = inputs.get_shape().as_list()
    net = tf.reshape(inputs, [-1, height, width, filters])
    for conv_id in range(4):
        net = self._conv_stage1[conv_id](net)
    net = self._conv_stage2(net)
    mask_outputs = self._conv_stage3(net)
    mask_outputs = tf.reshape(mask_outputs, [-1, num_rois, self.
        _mrcnn_resolution, self._mrcnn_resolution, self._num_classes])
    with tf.name_scope('masks_post_processing'):
        mask_outputs = tf.transpose(a=mask_outputs, perm=[0, 1, 4, 2, 3])
        indices_dtype = tf.float32 if self._is_gpu_inference else tf.int32
        class_indices = tf.cast(class_indices, indices_dtype)
        if batch_size == 1:
            indices = tf.reshape(tf.reshape(tf.range(num_rois, dtype=
                indices_dtype), [batch_size, num_rois, 1]) * self.
                _num_classes + tf.expand_dims(class_indices, axis=-1), [
                batch_size, -1])
            indices = tf.cast(indices, tf.int32)
            mask_outputs = tf.gather(tf.reshape(mask_outputs, [batch_size, 
                -1, self._mrcnn_resolution, self._mrcnn_resolution]),
                indices, axis=1)
            mask_outputs = tf.squeeze(mask_outputs, axis=1)
            mask_outputs = tf.reshape(mask_outputs, [batch_size, num_rois,
                self._mrcnn_resolution, self._mrcnn_resolution])
        else:
            batch_indices = tf.expand_dims(tf.range(batch_size, dtype=
                indices_dtype), axis=1) * tf.ones([1, num_rois], dtype=
                indices_dtype)
            mask_indices = tf.expand_dims(tf.range(num_rois, dtype=
                indices_dtype), axis=0) * tf.ones([batch_size, 1], dtype=
                indices_dtype)
            gather_indices = tf.stack([batch_indices, mask_indices,
                class_indices], axis=2)
            if self._is_gpu_inference:
                gather_indices = tf.cast(gather_indices, dtype=tf.int32)
            mask_outputs = tf.gather_nd(mask_outputs, gather_indices)
    return mask_outputs

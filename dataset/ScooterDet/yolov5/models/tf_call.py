def call(self, input, topk_all, iou_thres, conf_thres):
    return tf.map_fn(lambda x: self._nms(x, topk_all, iou_thres, conf_thres
        ), input, fn_output_signature=(tf.float32, tf.float32, tf.float32,
        tf.int32), name='agnostic_nms')

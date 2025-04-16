def encode_additional_info(self, predict_net, init_net):
    size_divisibility = self._wrapped_model.backbone.size_divisibility
    check_set_pb_arg(predict_net, 'size_divisibility', 'i', size_divisibility)
    check_set_pb_arg(predict_net, 'device', 's', str.encode(str(self.
        _wrapped_model.device), 'ascii'))
    check_set_pb_arg(predict_net, 'meta_architecture', 's', b'RetinaNet')
    check_set_pb_arg(predict_net, 'score_threshold', 'f', _cast_to_f32(self
        ._wrapped_model.test_score_thresh))
    check_set_pb_arg(predict_net, 'topk_candidates', 'i', self.
        _wrapped_model.test_topk_candidates)
    check_set_pb_arg(predict_net, 'nms_threshold', 'f', _cast_to_f32(self.
        _wrapped_model.test_nms_thresh))
    check_set_pb_arg(predict_net, 'max_detections_per_image', 'i', self.
        _wrapped_model.max_detections_per_image)
    check_set_pb_arg(predict_net, 'bbox_reg_weights', 'floats', [
        _cast_to_f32(w) for w in self._wrapped_model.box2box_transform.weights]
        )
    self._encode_anchor_generator_cfg(predict_net)

@staticmethod
def get_outputs_converter(predict_net, init_net):
    self = types.SimpleNamespace()
    serialized_anchor_generator = io.BytesIO(get_pb_arg_vals(predict_net,
        'serialized_anchor_generator', None))
    self.anchor_generator = torch.load(serialized_anchor_generator)
    bbox_reg_weights = get_pb_arg_floats(predict_net, 'bbox_reg_weights', None)
    self.box2box_transform = Box2BoxTransform(weights=tuple(bbox_reg_weights))
    self.test_score_thresh = get_pb_arg_valf(predict_net, 'score_threshold',
        None)
    self.test_topk_candidates = get_pb_arg_vali(predict_net,
        'topk_candidates', None)
    self.test_nms_thresh = get_pb_arg_valf(predict_net, 'nms_threshold', None)
    self.max_detections_per_image = get_pb_arg_vali(predict_net,
        'max_detections_per_image', None)
    for meth in ['forward_inference', 'inference_single_image',
        '_transpose_dense_predictions', '_decode_multi_level_predictions',
        '_decode_per_level_predictions']:
        setattr(self, meth, functools.partial(getattr(meta_arch.RetinaNet,
            meth), self))

    def f(batched_inputs, c2_inputs, c2_results):
        _, im_info = c2_inputs
        image_sizes = [[int(im[0]), int(im[1])] for im in im_info]
        dummy_images = ImageList(torch.randn((len(im_info), 3) + tuple(
            image_sizes[0])), image_sizes)
        num_features = len([x for x in c2_results.keys() if x.startswith(
            'box_cls_')])
        pred_logits = [c2_results['box_cls_{}'.format(i)] for i in range(
            num_features)]
        pred_anchor_deltas = [c2_results['box_delta_{}'.format(i)] for i in
            range(num_features)]
        dummy_features = [x.clone()[:, 0:0, :, :] for x in pred_logits]
        self.num_classes = pred_logits[0].shape[1] // (pred_anchor_deltas[0
            ].shape[1] // 4)
        results = self.forward_inference(dummy_images, dummy_features, [
            pred_logits, pred_anchor_deltas])
        return meta_arch.GeneralizedRCNN._postprocess(results,
            batched_inputs, image_sizes)
    return f

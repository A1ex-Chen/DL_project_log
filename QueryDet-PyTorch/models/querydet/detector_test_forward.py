def test_forward(self, images):
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    start_event.record()
    features = self.backbone(images.tensor[:, :, :])
    all_features = [features[f] for f in self.in_features]
    all_anchors, all_centers = self.anchor_generator(all_features)
    features_whole = [all_features[x] for x in self.layers_whole_test]
    features_value = [all_features[x] for x in self.layers_value_test]
    features_key = [all_features[x] for x in self.query_layer_test]
    anchors_whole = [all_anchors[x] for x in self.layers_whole_test]
    anchors_value = [all_anchors[x] for x in self.layers_value_test]
    det_cls_whole, det_delta_whole = self.det_head(features_whole)
    if not self.query_infer:
        det_cls_query, det_bbox_query = self.det_head(features_value)
        det_cls_query = [permute_to_N_HWA_K(x, self.num_classes) for x in
            det_cls_query]
        det_bbox_query = [permute_to_N_HWA_K(x, 4) for x in det_bbox_query]
        query_anchors = anchors_value
    else:
        if not self.qInfer.initialized:
            cls_weights, cls_biases, bbox_weights, bbox_biases = (self.
                det_head.get_params())
            qcls_weights, qcls_bias = self.query_head.get_params()
            params = [cls_weights, cls_biases, bbox_weights, bbox_biases,
                qcls_weights, qcls_bias]
        else:
            params = None
        det_cls_query, det_bbox_query, query_anchors = self.qInfer.run_qinfer(
            params, features_key, features_value, anchors_value)
    results = self.inference(det_cls_whole, det_delta_whole, anchors_whole,
        det_cls_query, det_bbox_query, query_anchors, images.image_sizes)
    end_event.record()
    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)
    return results, total_time

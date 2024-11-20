def run_qinfer(self, model_params, features_key, features_value, anchors_value
    ):
    if not self.initialized:
        (cls_weights, cls_biases, bbox_weights, bbox_biases, qcls_weights,
            qcls_biases) = model_params
        assert len(cls_weights) == len(qcls_weights)
        self.n_conv = len(cls_weights)
        self.cls_spconv = self._make_spconv(cls_weights, cls_biases)
        self.bbox_spconv = self._make_spconv(bbox_weights, bbox_biases)
        self.qcls_spconv = self._make_spconv(qcls_weights, qcls_biases)
        self.qcls_conv = self._make_conv(qcls_weights, qcls_biases)
        self.initialized = True
    last_ys, last_xs = None, None
    query_logits = self._run_convs(features_key[-1], self.qcls_conv)
    det_cls_query, det_bbox_query, query_anchors = [], [], []
    n_inds_all = []
    for i in range(len(features_value) - 1, -1, -1):
        x, last_ys, last_xs, inds, selected_anchors, n_inds = (self.
            _make_sparse_tensor(query_logits, last_ys, last_xs,
            anchors_value[i], features_value[i]))
        n_inds_all.append(n_inds)
        if x == None:
            break
        cls_result = self._run_spconvs(x, self.cls_spconv).view(-1, self.
            anchor_num * self.num_classes)[inds]
        bbox_result = self._run_spconvs(x, self.bbox_spconv).view(-1, self.
            anchor_num * 4)[inds]
        query_logits = self._run_spconvs(x, self.qcls_spconv).view(-1)[inds]
        query_anchors.append(selected_anchors)
        det_cls_query.append(torch.unsqueeze(cls_result, 0))
        det_bbox_query.append(torch.unsqueeze(bbox_result, 0))
    return det_cls_query, det_bbox_query, query_anchors

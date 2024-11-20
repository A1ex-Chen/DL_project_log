def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
    """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
    box_features = box_features.permute(0, 2, 3, 1)
    num_layers, batch, channel, num_queries = box_features.shape[0
        ], box_features.shape[1], box_features.shape[2], box_features.shape[3]
    box_features = box_features.reshape(num_layers * batch, channel,
        num_queries)
    cls_logits = self.mlp_heads['sem_cls_head'](box_features).transpose(1, 2)
    center_offset = self.mlp_heads['center_head'](box_features).sigmoid(
        ).transpose(1, 2) - 0.5
    size_normalized = self.mlp_heads['size_head'](box_features).sigmoid(
        ).transpose(1, 2)
    angle_logits = self.mlp_heads['angle_cls_head'](box_features).transpose(
        1, 2)
    angle_residual_normalized = self.mlp_heads['angle_residual_head'](
        box_features).transpose(1, 2)
    cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
    center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
    size_normalized = size_normalized.reshape(num_layers, batch,
        num_queries, -1)
    angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
    angle_residual_normalized = angle_residual_normalized.reshape(num_layers,
        batch, num_queries, -1)
    angle_residual = angle_residual_normalized * (np.pi /
        angle_residual_normalized.shape[-1])
    outputs = []
    for l in range(num_layers):
        center_normalized, center_unnormalized = (self.box_processor.
            compute_predicted_center(center_offset[l], query_xyz,
            point_cloud_dims))
        angle_continuous = self.box_processor.compute_predicted_angle(
            angle_logits[l], angle_residual[l])
        size_unnormalized = self.box_processor.compute_predicted_size(
            size_normalized[l], point_cloud_dims)
        box_corners = self.box_processor.box_parametrization_to_corners(
            center_unnormalized, size_unnormalized, angle_continuous)
        with torch.no_grad():
            semcls_prob, objectness_prob = (self.box_processor.
                compute_objectness_and_cls_prob(cls_logits[l]))
        box_prediction = {'sem_cls_logits': cls_logits[l],
            'center_normalized': center_normalized.contiguous(),
            'center_unnormalized': center_unnormalized, 'size_normalized':
            size_normalized[l], 'size_unnormalized': size_unnormalized,
            'angle_logits': angle_logits[l], 'angle_residual':
            angle_residual[l], 'angle_residual_normalized':
            angle_residual_normalized[l], 'angle_continuous':
            angle_continuous, 'objectness_prob': objectness_prob,
            'sem_cls_prob': semcls_prob, 'box_corners': box_corners}
        outputs.append(box_prediction)
    aux_outputs = outputs[:-1]
    outputs = outputs[-1]
    return {'outputs': outputs, 'aux_outputs': aux_outputs}

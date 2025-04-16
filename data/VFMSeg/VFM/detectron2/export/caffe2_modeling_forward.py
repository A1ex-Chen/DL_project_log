@mock_torch_nn_functional_interpolate()
def forward(self, inputs):
    assert self.tensor_mode
    images = self._caffe2_preprocess_image(inputs)
    return_tensors = [images.image_sizes]
    features = self._wrapped_model.backbone(images.tensor)
    features = [features[f] for f in self._wrapped_model.head_in_features]
    for i, feature_i in enumerate(features):
        features[i] = alias(feature_i, 'feature_{}'.format(i), is_backward=True
            )
        return_tensors.append(features[i])
    pred_logits, pred_anchor_deltas = self._wrapped_model.head(features)
    for i, (box_cls_i, box_delta_i) in enumerate(zip(pred_logits,
        pred_anchor_deltas)):
        return_tensors.append(alias(box_cls_i, 'box_cls_{}'.format(i)))
        return_tensors.append(alias(box_delta_i, 'box_delta_{}'.format(i)))
    return tuple(return_tensors)

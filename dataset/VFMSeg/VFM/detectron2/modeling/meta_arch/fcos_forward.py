def forward(self, features):
    assert len(features) == self._num_features
    logits = []
    bbox_reg = []
    ctrness = []
    for feature in features:
        logits.append(self.cls_score(self.cls_subnet(feature)))
        bbox_feature = self.bbox_subnet(feature)
        bbox_reg.append(self.bbox_pred(bbox_feature))
        ctrness.append(self.ctrness(bbox_feature))
    return logits, bbox_reg, ctrness

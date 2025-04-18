def forward(self, images: ImageList, features: Dict[str, torch.Tensor],
    gt_instances: Optional[List[Instances]]=None):
    """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
    features = [features[f] for f in self.in_features]
    anchors = self.anchor_generator(features)
    pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
    pred_objectness_logits = [score.permute(0, 2, 3, 1).flatten(1) for
        score in pred_objectness_logits]
    pred_anchor_deltas = [x.view(x.shape[0], -1, self.anchor_generator.
        box_dim, x.shape[-2], x.shape[-1]).permute(0, 3, 4, 1, 2).flatten(1,
        -2) for x in pred_anchor_deltas]
    if self.training:
        assert gt_instances is not None, 'RPN requires gt_instances in training!'
        gt_labels, gt_boxes = self.label_and_sample_anchors(anchors,
            gt_instances)
        losses = self.losses(anchors, pred_objectness_logits, gt_labels,
            pred_anchor_deltas, gt_boxes)
    else:
        losses = {}
    proposals = self.predict_proposals(anchors, pred_objectness_logits,
        pred_anchor_deltas, images.image_sizes)
    return proposals, losses

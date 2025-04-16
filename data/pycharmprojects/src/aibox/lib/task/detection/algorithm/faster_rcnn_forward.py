def forward(self, padded_image_batch: Tensor, gt_bboxes_batch: List[Tensor]
    =None, gt_classes_batch: List[Tensor]=None) ->Union[Tuple[Tensor,
    Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor],
    List[Tensor], List[Tensor], List[Tensor]]]:
    batch_size, _, padded_image_height, padded_image_width = (
        padded_image_batch.shape)
    features_batch = self.body(padded_image_batch)
    _, _, features_height, features_width = features_batch.shape
    anchor_objectnesses_batch, anchor_transformers_batch = (self.rpn_head.
        forward(features_batch))
    anchor_bboxes = self.rpn_head.generate_anchors(padded_image_width,
        padded_image_height, num_x_anchors=features_width, num_y_anchors=
        features_height, scale=1.0).to(features_batch)
    anchor_bboxes_batch: List[Tensor] = [anchor_bboxes] * batch_size
    proposal_bboxes_batch, proposal_probs_batch = (self.rpn_head.
        generate_proposals_batch(anchor_bboxes_batch,
        anchor_objectnesses_batch, anchor_transformers_batch,
        padded_image_width, padded_image_height))
    if self.training:
        for b in range(batch_size):
            gt_bboxes = gt_bboxes_batch[b]
            gt_classes = gt_classes_batch[b]
            if gt_bboxes.shape[0] == 0 and gt_classes.shape[0] == 0:
                gt_bboxes_batch[b] = torch.tensor([[padded_image_width // 4,
                    padded_image_height // 4, padded_image_width // 4 + 
                    padded_image_width // 2, padded_image_height // 4 + 
                    padded_image_height // 2]]).to(gt_bboxes)
                gt_classes_batch[b] = torch.tensor([0]).to(gt_classes)
        anchor_objectness_loss_batch = torch.zeros((batch_size,)).to(
            features_batch)
        anchor_transformer_loss_batch = torch.zeros((batch_size,)).to(
            features_batch)
        for b in range(batch_size):
            gt_bboxes = gt_bboxes_batch[b]
            gt_classes = gt_classes_batch[b]
            anchor_bboxes = anchor_bboxes_batch[b]
            anchor_objectnesses = anchor_objectnesses_batch[b]
            anchor_transformers = anchor_transformers_batch[b]
            (sampled_indices, sampled_gt_anchor_objectnesses,
                sampled_gt_anchor_transformers) = (self.rpn_head.sample(
                anchor_bboxes, gt_bboxes, gt_classes, padded_image_width,
                padded_image_height))
            if sampled_indices.shape[0] == 0:
                continue
            sampled_anchor_objectnesses = anchor_objectnesses[sampled_indices]
            sampled_anchor_transformers = anchor_transformers[sampled_indices]
            anchor_objectness_loss, anchor_transformer_loss = (self.
                rpn_head.loss(sampled_anchor_objectnesses,
                sampled_anchor_transformers, sampled_gt_anchor_objectnesses,
                sampled_gt_anchor_transformers))
            anchor_objectness_loss_batch[b] = anchor_objectness_loss
            anchor_transformer_loss_batch[b] = anchor_transformer_loss
        proposal_class_loss_batch = torch.zeros((batch_size,)).to(
            features_batch)
        proposal_transformer_loss_batch = torch.zeros((batch_size,)).to(
            features_batch)
        for b in range(batch_size):
            features = features_batch[b]
            gt_bboxes = gt_bboxes_batch[b]
            gt_classes = gt_classes_batch[b]
            proposal_bboxes = proposal_bboxes_batch[b]
            (sampled_indices, sampled_gt_proposal_classes,
                sampled_gt_proposal_transformers) = (self.roi_head.sample(
                proposal_bboxes, gt_bboxes, gt_classes))
            sampled_proposal_bboxes = proposal_bboxes[sampled_indices]
            if sampled_proposal_bboxes.shape[0] == 0:
                continue
            pools = self._roi_align(input=features.unsqueeze(dim=0), rois=[
                sampled_proposal_bboxes])
            pools = F.max_pool2d(input=pools, kernel_size=2, stride=2)
            proposal_classes, proposal_transformers = self.roi_head.forward(
                pools, post_extract_transform=lambda x: F.
                adaptive_avg_pool2d(input=x, output_size=1).view(x.shape[0],
                -1))
            proposal_class_loss, proposal_transformer_loss = (self.roi_head
                .loss(proposal_classes, proposal_transformers,
                sampled_gt_proposal_classes, sampled_gt_proposal_transformers))
            proposal_class_loss_batch[b] = proposal_class_loss
            proposal_transformer_loss_batch[b] = proposal_transformer_loss
        return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
            proposal_class_loss_batch, proposal_transformer_loss_batch)
    else:
        pools_batch = []
        for b in range(batch_size):
            features = features_batch[b]
            proposal_bboxes = proposal_bboxes_batch[b]
            pools = self._roi_align(input=features.unsqueeze(dim=0), rois=[
                proposal_bboxes])
            pools = F.max_pool2d(input=pools, kernel_size=2, stride=2)
            pools_batch.append(pools)
        (detection_bboxes_batch, detection_classes_batch, detection_probs_batch
            ) = [], [], []
        for pools in pools_batch:
            proposal_classes, proposal_transformers = self.roi_head.forward(
                pools, post_extract_transform=lambda x: F.
                adaptive_avg_pool2d(input=x, output_size=1).view(x.shape[0],
                -1))
            detection_bboxes, detection_classes, detection_probs = (self.
                roi_head.generate_detections(proposal_bboxes,
                proposal_classes, proposal_transformers, padded_image_width,
                padded_image_height))
            detection_bboxes_batch.append(detection_bboxes)
            detection_classes_batch.append(detection_classes)
            detection_probs_batch.append(detection_probs)
        return (anchor_bboxes_batch, proposal_bboxes_batch,
            proposal_probs_batch, detection_bboxes_batch,
            detection_classes_batch, detection_probs_batch)

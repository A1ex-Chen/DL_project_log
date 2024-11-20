def forward(self, image_batch: Bunch, gt_bboxes_batch: Bunch=None,
    gt_classes_batch: Bunch=None) ->Union[Tuple[Bunch, Bunch, Bunch, Bunch],
    Tuple[Bunch, Bunch, Bunch, Bunch, Bunch, Bunch]]:
    batch_size = len(image_batch)
    if batch_size == 1:
        padded_image_batch = image_batch[0].unsqueeze(dim=0)
    else:
        padded_image_width = max([it.shape[2] for it in image_batch])
        padded_image_height = max([it.shape[1] for it in image_batch])
        padded_image_batch = []
        for image in image_batch:
            padded_image = F.pad(input=image, pad=[0, padded_image_width -
                image.shape[2], 0, padded_image_height - image.shape[1]])
            padded_image_batch.append(padded_image)
        padded_image_batch = torch.stack(padded_image_batch, dim=0)
    padded_image_batch = normalize_means_stds(padded_image_batch, list(self
        .algorithm.backbone.normalization_means()), list(self.algorithm.
        backbone.normalization_stds()))
    if self.training:
        (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
            proposal_class_loss_batch, proposal_transformer_loss_batch) = (self
            .algorithm.forward(padded_image_batch, gt_bboxes_batch.tolist(),
            gt_classes_batch.tolist()))
        anchor_objectness_loss_batch = Bunch(anchor_objectness_loss_batch.
            unbind(dim=0))
        anchor_transformer_loss_batch = Bunch(anchor_transformer_loss_batch
            .unbind(dim=0))
        proposal_class_loss_batch = Bunch(proposal_class_loss_batch.unbind(
            dim=0))
        proposal_transformer_loss_batch = Bunch(proposal_transformer_loss_batch
            .unbind(dim=0))
        return (anchor_objectness_loss_batch, anchor_transformer_loss_batch,
            proposal_class_loss_batch, proposal_transformer_loss_batch)
    else:
        (anchor_bboxes_batch, proposal_bboxes_batch, proposal_probs_batch,
            detection_bboxes_batch, detection_classes_batch,
            detection_probs_batch) = self.algorithm.forward(padded_image_batch)
        anchor_bboxes_batch = Bunch(anchor_bboxes_batch)
        proposal_bboxes_batch = Bunch(proposal_bboxes_batch)
        proposal_probs_batch = Bunch(proposal_probs_batch)
        detection_bboxes_batch = Bunch(detection_bboxes_batch)
        detection_classes_batch = Bunch(detection_classes_batch)
        detection_probs_batch = Bunch(detection_probs_batch)
        return (anchor_bboxes_batch, proposal_bboxes_batch,
            proposal_probs_batch, detection_bboxes_batch,
            detection_classes_batch, detection_probs_batch)

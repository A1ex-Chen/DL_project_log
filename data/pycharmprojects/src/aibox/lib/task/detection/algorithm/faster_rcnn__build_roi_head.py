def _build_roi_head(self) ->ROI:
    num_extractor_out = self.backbone.component.num_conv5_out
    extractor = self.backbone.component.conv5
    head = ROI(extractor, num_extractor_out, self.num_classes, self.
        num_proposal_samples_per_batch, self.num_detections_per_image, self
        .proposal_smooth_l1_loss_beta, self.detection_nms_threshold)
    return head

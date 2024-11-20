def _build_roi_head(self, num_extractor_in: int) ->ROI:
    num_extractor_out = 1024
    extractor = nn.Sequential(nn.Linear(in_features=num_extractor_in * 7 * 
        7, out_features=num_extractor_out), nn.ReLU(), nn.Linear(
        in_features=num_extractor_out, out_features=num_extractor_out), nn.
        ReLU())
    head = ROI(extractor, num_extractor_out, self.num_classes, self.
        num_proposal_samples_per_batch, self.num_detections_per_image, self
        .proposal_smooth_l1_loss_beta, self.detection_nms_threshold)
    for m in head.children():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1)
            nn.init.constant_(m.bias, val=0)
    return head

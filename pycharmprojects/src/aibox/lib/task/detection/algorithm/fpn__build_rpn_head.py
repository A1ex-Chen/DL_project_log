def _build_rpn_head(self, num_extractor_in: int) ->Tuple[RPN, int]:
    num_extractor_out = 256
    extractor = nn.Sequential(nn.Conv2d(in_channels=num_extractor_in,
        out_channels=num_extractor_out, kernel_size=3, padding=1), nn.ReLU())
    head = RPN(extractor, num_extractor_out, self.anchor_ratios, self.
        anchor_sizes, self.train_rpn_pre_nms_top_n, self.
        train_rpn_post_nms_top_n, self.eval_rpn_pre_nms_top_n, self.
        eval_rpn_post_nms_top_n, self.num_anchor_samples_per_batch, self.
        anchor_smooth_l1_loss_beta, self.proposal_nms_threshold)
    for m in head.children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, val=0)
    return head, num_extractor_out

def loss(self, proposal_classes: Tensor, proposal_transformers: Tensor,
    gt_proposal_classes: Tensor, gt_proposal_transformers: Tensor) ->Tuple[
    Tensor, Tensor]:
    proposal_transformers = proposal_transformers[torch.arange(end=
        proposal_transformers.shape[0]), gt_proposal_classes]
    transformer_normalize_mean = self._transformer_normalize_mean.to(device
        =gt_proposal_transformers.device)
    transformer_normalize_std = self._transformer_normalize_std.to(device=
        gt_proposal_transformers.device)
    gt_proposal_transformers = (gt_proposal_transformers -
        transformer_normalize_mean) / transformer_normalize_std
    proposal_class_loss = F.cross_entropy(input=proposal_classes, target=
        gt_proposal_classes)
    fg_indices = gt_proposal_classes.nonzero().flatten()
    proposal_transformer_loss = beta_smooth_l1_loss(input=
        proposal_transformers[fg_indices], target=gt_proposal_transformers[
        fg_indices], beta=self._proposal_smooth_l1_loss_beta)
    return proposal_class_loss, proposal_transformer_loss

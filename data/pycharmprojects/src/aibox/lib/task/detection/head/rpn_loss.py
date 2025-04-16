def loss(self, anchor_objectnesses: Tensor, anchor_transformers: Tensor,
    gt_anchor_objectnesses: Tensor, gt_anchor_transformers: Tensor) ->Tuple[
    Tensor, Tensor]:
    cross_entropy = F.cross_entropy(input=anchor_objectnesses, target=
        gt_anchor_objectnesses)
    fg_indices = gt_anchor_objectnesses.nonzero().flatten()
    smooth_l1_loss = beta_smooth_l1_loss(input=anchor_transformers[
        fg_indices], target=gt_anchor_transformers[fg_indices], beta=self.
        _anchor_smooth_l1_loss_beta)
    return cross_entropy, smooth_l1_loss

def _reduce_pred_masks(self, outputs, tfms):
    for output, tfm in zip(outputs, tfms):
        if any(isinstance(t, HFlipTransform) for t in tfm.transforms):
            output.pred_masks = output.pred_masks.flip(dims=[3])
    all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
    avg_pred_masks = torch.mean(all_pred_masks, dim=0)
    return avg_pred_masks

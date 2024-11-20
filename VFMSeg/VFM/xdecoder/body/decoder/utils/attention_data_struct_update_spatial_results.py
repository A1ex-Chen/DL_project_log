def update_spatial_results(self, results):
    v_emb = results['pred_smaskembs']
    pred_smasks = results['pred_smasks']
    s_emb = results['pred_pspatials']
    pred_logits = v_emb @ s_emb.transpose(1, 2)
    logits_idx_y = pred_logits[:, :, 0].max(dim=1)[1]
    logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
    logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
    pred_masks_pos = pred_smasks[logits_idx][:, None]
    extra = {'prev_mask': pred_masks_pos}
    return extra

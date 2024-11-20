def forward_prediction_heads(self, output, mask_features,
    attn_mask_target_size, layer_id=-1):
    decoder_output = self.decoder_norm(output)
    decoder_output = decoder_output.transpose(0, 1)
    class_embed = decoder_output @ self.class_embed
    outputs_class = self.lang_encoder.compute_similarity(class_embed)
    mask_embed = self.mask_embed(decoder_output)
    outputs_mask = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
    outputs_bbox = [None for i in range(len(outputs_mask))]
    if self.task_switch['bbox']:
        outputs_bbox = self.bbox_embed(decoder_output)
    attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size,
        mode='bilinear', align_corners=False)
    attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self
        .num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
    attn_mask = attn_mask.detach()
    outputs_caption = class_embed
    results = {'attn_mask': attn_mask, 'predictions_class': outputs_class,
        'predictions_mask': outputs_mask, 'predictions_bbox': outputs_bbox,
        'predictions_caption': outputs_caption, 'predictions_maskemb':
        mask_embed}
    return results

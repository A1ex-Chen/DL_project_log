def forward(self, x, batch=None):
    """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
    from ultralytics.models.utils.ops import get_cdn_group
    feats, shapes = self._get_encoder_input(x)
    dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(batch, self.nc,
        self.num_queries, self.denoising_class_embed.weight, self.
        num_denoising, self.label_noise_ratio, self.box_noise_scale, self.
        training)
    embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats,
        shapes, dn_embed, dn_bbox)
    dec_bboxes, dec_scores = self.decoder(embed, refer_bbox, feats, shapes,
        self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
        attn_mask=attn_mask)
    x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
    if self.training:
        return x
    y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
    return y if self.export else (y, x)

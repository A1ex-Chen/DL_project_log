def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
    """Generates and prepares the input required for the decoder from the provided features and shapes."""
    bs = feats.shape[0]
    anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype,
        device=feats.device)
    features = self.enc_output(valid_mask * feats)
    enc_outputs_scores = self.enc_score_head(features)
    topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.
        num_queries, dim=1).indices.view(-1)
    batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1
        ).repeat(1, self.num_queries).view(-1)
    top_k_features = features[batch_ind, topk_ind].view(bs, self.
        num_queries, -1)
    top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)
    refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors
    enc_bboxes = refer_bbox.sigmoid()
    if dn_bbox is not None:
        refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
    enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.
        num_queries, -1)
    embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1
        ) if self.learnt_init_query else top_k_features
    if self.training:
        refer_bbox = refer_bbox.detach()
        if not self.learnt_init_query:
            embeddings = embeddings.detach()
    if dn_embed is not None:
        embeddings = torch.cat([dn_embed, embeddings], 1)
    return embeddings, refer_bbox, enc_bboxes, enc_scores

def forward(input_ids, segment_ids, input_mask, input_ent_neighbor_emb,
    ent_mask, label_ids):
    logits_ernie, loss_ernie = self.model_ernie(input_ids, segment_ids,
        input_mask, input_ent_neighbor_emb, ent_mask, label_ids)
    loss = self.model_att(input_ids, segment_ids, input_mask, input_ent,
        ent_mask, label_ids, k, v, logits_ernie)
    return logits_ernie, loss_ernie, loss

def _reset_parameters(self):
    """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
    bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
    constant_(self.enc_score_head.bias, bias_cls)
    constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
    constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
    for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
        constant_(cls_.bias, bias_cls)
        constant_(reg_.layers[-1].weight, 0.0)
        constant_(reg_.layers[-1].bias, 0.0)
    linear_init(self.enc_output[0])
    xavier_uniform_(self.enc_output[0].weight)
    if self.learnt_init_query:
        xavier_uniform_(self.tgt_embed.weight)
    xavier_uniform_(self.query_pos_head.layers[0].weight)
    xavier_uniform_(self.query_pos_head.layers[1].weight)
    for layer in self.input_proj:
        xavier_uniform_(layer[0].weight)

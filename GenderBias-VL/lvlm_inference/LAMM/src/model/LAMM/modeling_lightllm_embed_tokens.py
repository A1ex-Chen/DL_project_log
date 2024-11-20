def embed_tokens(self, input_ids):
    embed_tokens = self.base_model.pre_infer.token_forward(input_ids=
        input_ids, infer_state=None, layer_weight=self.base_model.
        pre_post_weight)
    return embed_tokens

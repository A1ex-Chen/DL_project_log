def get_pos_embedding(self, vision_input, all_vision_tokens):
    input_shape = vision_input.shape
    pos_embed = _get_pos_embedding(all_vision_tokens.size(1) - self.
        num_cls_tokens, pos_embed=self.pos_embed, patches_layout=self.
        patches_layout, input_shape=input_shape, first_patch_idx=self.
        num_cls_tokens)
    return pos_embed

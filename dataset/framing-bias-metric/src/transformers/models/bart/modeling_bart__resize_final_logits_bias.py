def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int
    ) ->None:
    if new_num_tokens <= old_num_tokens:
        new_bias = self.final_logits_bias[:, :new_num_tokens]
    else:
        extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens),
            device=self.final_logits_bias.device)
        new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
    self.register_buffer('final_logits_bias', new_bias)

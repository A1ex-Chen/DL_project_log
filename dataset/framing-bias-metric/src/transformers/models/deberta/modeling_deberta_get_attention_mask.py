def get_attention_mask(self, attention_mask):
    if attention_mask.dim() <= 2:
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (extended_attention_mask * extended_attention_mask
            .squeeze(-2).unsqueeze(-1))
        attention_mask = attention_mask.byte()
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)
    return attention_mask

@torch.no_grad()
@eval_decorator
def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0,
    filter_thres=0.9, **kwargs):
    """
        Generates autoregressive sequences based on the given start tokens.

        Args:
            start_tokens (torch.Tensor): The initial tokens to start the generation.
            seq_len (int): The length of the generated sequence.
            eos_token (int, optional): The end-of-sequence token. If provided, generation will stop when this token is generated. Defaults to None.
            temperature (float, optional): The temperature value for controlling the randomness of the generation. Higher values result in more randomness. Defaults to 1.0.
            filter_thres (float, optional): The threshold value for filtering logits during generation. Only logits above this threshold will be considered. Defaults to 0.9.
            **kwargs: Additional keyword arguments to be passed to the underlying network.

        Returns:
            torch.Tensor: The generated sequence.
        """
    b, t, device = *start_tokens.shape, start_tokens.device
    out = start_tokens
    for _ in range(seq_len):
        logits = self.net(out, **kwargs)[:, -1, :]
        filtered_logits = top_k(logits, thres=filter_thres)
        probs = F.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)
        out = torch.cat((out, sample), dim=-1)
        if exists(eos_token):
            is_eos_token = out == eos_token
            if is_eos_token.any(dim=-1).all():
                shifted_is_eos_tokens = F.pad(is_eos_token, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = out.masked_fill(mask, self.pad_value)
                break
    out = out[:, t:]
    return out

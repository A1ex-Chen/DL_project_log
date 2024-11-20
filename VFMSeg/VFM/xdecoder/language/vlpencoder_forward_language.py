def forward_language(self, texts, norm=True):
    x = self.lang_encoder(*texts)
    x = x['last_hidden_state']
    if self.tokenizer_type == 'clip':
        x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
    else:
        x = x[:, 0]
    x = x @ self.lang_proj
    if norm:
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-07)
    return x

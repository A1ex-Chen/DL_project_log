def forward_language_token(self, texts, norm=False):
    x = self.lang_encoder(*texts)
    token_x = x['last_hidden_state']
    if self.tokenizer_type == 'clip':
        class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(
            dim=-1)]
    else:
        class_x = token_x[:, 0]
    class_x = class_x @ self.lang_proj
    token_x = token_x @ self.lang_proj
    if norm:
        class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-07)
        token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-07)
    return token_x, class_x

def encode_text(self, text):
    x = self.token_embedding(text).type(self.dtype)
    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = self.ln_final(x).type(self.dtype)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x

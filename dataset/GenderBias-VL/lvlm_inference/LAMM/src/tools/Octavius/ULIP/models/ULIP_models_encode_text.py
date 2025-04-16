def encode_text(self, text):
    x = self.token_embedding(text)
    x = x + self.positional_embedding
    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = self.ln_final(x)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x

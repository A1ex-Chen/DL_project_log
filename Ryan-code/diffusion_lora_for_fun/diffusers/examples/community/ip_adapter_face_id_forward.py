def forward(self, image_embeds: torch.Tensor):
    x = self.ff(image_embeds)
    x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
    return self.norm(x)

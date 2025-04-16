def forward(self, features, **kwargs):
    x = self.dense(features)
    x = x.view(-1, kwargs['src_len'], x.size(-1)).transpose(0, 1)
    bsz = x.size(1)
    latent_query = self.latent_query.unsqueeze(1).expand(-1, bsz, -1)
    x, _ = self.x_attn(latent_query, torch.cat([x, latent_query]), torch.
        cat([x, latent_query]))
    return x.transpose(0, 1).contiguous().view(-1, x.size(-1))

def gen_r_embedding(self, r, max_positions=10000):
    r = r * max_positions
    half_dim = self.c_r // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
    emb = r[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if self.c_r % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1), mode='constant')
    return emb.to(dtype=r.dtype)

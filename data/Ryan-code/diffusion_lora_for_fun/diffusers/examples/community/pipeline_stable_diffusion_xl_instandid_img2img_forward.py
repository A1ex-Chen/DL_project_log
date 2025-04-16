def forward(self, x):
    latents = self.latents.repeat(x.size(0), 1, 1)
    x = self.proj_in(x)
    for attn, ff in self.layers:
        latents = attn(x, latents) + latents
        latents = ff(latents) + latents
    latents = self.proj_out(latents)
    return self.norm_out(latents)

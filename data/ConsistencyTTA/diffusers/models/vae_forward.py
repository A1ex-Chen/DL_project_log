def forward(self, z):
    z = z.permute(0, 2, 3, 1).contiguous()
    z_flattened = z.view(-1, self.vq_embed_dim)
    min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.
        embedding.weight), dim=1)
    z_q = self.embedding(min_encoding_indices).view(z.shape)
    perplexity = None
    min_encodings = None
    if not self.legacy:
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2)
    else:
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2)
    z_q = z + (z_q - z).detach()
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    if self.remap is not None:
        min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
        min_encoding_indices = self.remap_to_used(min_encoding_indices)
        min_encoding_indices = min_encoding_indices.reshape(-1, 1)
    if self.sane_index_shape:
        min_encoding_indices = min_encoding_indices.reshape(z_q.shape[0],
            z_q.shape[2], z_q.shape[3])
    return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

def forward_features(self, x, longer_idx=None):
    frames_num = x.shape[2]
    x = self.patch_embed(x, longer_idx=longer_idx)
    if self.ape:
        x = x + self.absolute_pos_embed
    x = self.pos_drop(x)
    for i, layer in enumerate(self.layers):
        x, attn = layer(x)
    x = self.norm(x)
    B, N, C = x.shape
    SF = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[0]
    ST = frames_num // 2 ** (len(self.depths) - 1) // self.patch_stride[1]
    x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
    B, C, F, T = x.shape
    c_freq_bin = F // self.freq_ratio
    x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
    x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
    fine_grained_latent_output = torch.mean(x, dim=2)
    fine_grained_latent_output = interpolate(fine_grained_latent_output.
        permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
    latent_output = self.avgpool(torch.flatten(x, 2))
    latent_output = torch.flatten(latent_output, 1)
    x = self.tscam_conv(x)
    x = torch.flatten(x, 2)
    fpx = interpolate(torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 *
        self.patch_stride[1])
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    output_dict = {'framewise_output': fpx, 'clipwise_output': torch.
        sigmoid(x), 'fine_grained_embedding': fine_grained_latent_output,
        'embedding': latent_output}
    return output_dict

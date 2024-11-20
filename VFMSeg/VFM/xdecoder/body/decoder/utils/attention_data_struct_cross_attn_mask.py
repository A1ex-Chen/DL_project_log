def cross_attn_mask(self, size, num_heads):
    attn_mask = torch.cat([self.attn_variables[name].attn_mask for name in
        self.cross_attn_name], dim=1)
    if 'memories_spatial' in self.cross_attn_name:
        memory_attn_mask = self.spatial_memory['prev_batch_mask']
        bs, c, _, _ = memory_attn_mask.shape
        memory_attn_mask = F.interpolate(memory_attn_mask, size, mode=
            'bilinear', align_corners=False)
        memory_attn_mask = (memory_attn_mask.sigmoid().flatten(2).unsqueeze
            (1).repeat(1, num_heads, 1, 1).flatten(0, 1) < 0.5).bool().detach()
        attn_mask[:, self.query_index['memories_spatial'][0]:self.
            query_index['memories_spatial'][1]] = memory_attn_mask
    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
    return attn_mask

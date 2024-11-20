def get_position_embeds(self, seq_len, dtype, device):
    """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shif attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
    d_model = self.config.d_model
    if self.config.attention_type == 'factorized':
        pos_seq = torch.arange(0, seq_len, 1.0, dtype=dtype, device=device)
        freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=dtype, device=
            device)
        inv_freq = 1 / 10000 ** (freq_seq / (d_model // 2))
        sinusoid = pos_seq[:, None] * inv_freq[None]
        sin_embed = torch.sin(sinusoid)
        sin_embed_d = self.sin_dropout(sin_embed)
        cos_embed = torch.cos(sinusoid)
        cos_embed_d = self.cos_dropout(cos_embed)
        phi = torch.cat([sin_embed_d, sin_embed_d], dim=-1)
        psi = torch.cat([cos_embed, sin_embed], dim=-1)
        pi = torch.cat([cos_embed_d, cos_embed_d], dim=-1)
        omega = torch.cat([-sin_embed, cos_embed], dim=-1)
        return phi, pi, psi, omega
    else:
        freq_seq = torch.arange(0, d_model // 2, 1.0, dtype=dtype, device=
            device)
        inv_freq = 1 / 10000 ** (freq_seq / (d_model // 2))
        rel_pos_id = torch.arange(-seq_len * 2, seq_len * 2, 1.0, dtype=
            dtype, device=device)
        zero_offset = seq_len * 2
        sinusoid = rel_pos_id[:, None] * inv_freq[None]
        sin_embed = self.sin_dropout(torch.sin(sinusoid))
        cos_embed = self.cos_dropout(torch.cos(sinusoid))
        pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)
        pos = torch.arange(0, seq_len, dtype=dtype, device=device)
        pooled_pos = pos
        position_embeds_list = []
        for block_index in range(0, self.config.num_blocks):
            if block_index == 0:
                position_embeds_pooling = None
            else:
                pooled_pos = self.stride_pool_pos(pos, block_index)
                stride = 2 ** (block_index - 1)
                rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                rel_pos = rel_pos[:, None] + zero_offset
                rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
                position_embeds_pooling = torch.gather(pos_embed, 0, rel_pos)
            pos = pooled_pos
            stride = 2 ** block_index
            rel_pos = self.relative_pos(pos, stride)
            rel_pos = rel_pos[:, None] + zero_offset
            rel_pos = rel_pos.expand(rel_pos.size(0), d_model)
            position_embeds_no_pooling = torch.gather(pos_embed, 0, rel_pos)
            position_embeds_list.append([position_embeds_no_pooling,
                position_embeds_pooling])
        return position_embeds_list

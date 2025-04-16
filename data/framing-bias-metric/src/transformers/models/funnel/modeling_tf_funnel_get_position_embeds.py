def get_position_embeds(self, seq_len, dtype=tf.float32, training=False):
    """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shif attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
    if self.attention_type == 'factorized':
        pos_seq = tf.range(0, seq_len, 1.0, dtype=dtype)
        freq_seq = tf.range(0, self.d_model // 2, 1.0, dtype=dtype)
        inv_freq = 1 / 10000 ** (freq_seq / (self.d_model // 2))
        sinusoid = tf.einsum('i,d->id', pos_seq, inv_freq)
        sin_embed = tf.sin(sinusoid)
        sin_embed_d = self.sin_dropout(sin_embed, training=training)
        cos_embed = tf.cos(sinusoid)
        cos_embed_d = self.cos_dropout(cos_embed, training=training)
        phi = tf.concat([sin_embed_d, sin_embed_d], axis=-1)
        psi = tf.concat([cos_embed, sin_embed], axis=-1)
        pi = tf.concat([cos_embed_d, cos_embed_d], axis=-1)
        omega = tf.concat([-sin_embed, cos_embed], axis=-1)
        return phi, pi, psi, omega
    else:
        freq_seq = tf.range(0, self.d_model // 2, 1.0, dtype=dtype)
        inv_freq = 1 / 10000 ** (freq_seq / (self.d_model // 2))
        rel_pos_id = tf.range(-seq_len * 2, seq_len * 2, 1.0, dtype=dtype)
        zero_offset = seq_len * 2
        sinusoid = tf.einsum('i,d->id', rel_pos_id, inv_freq)
        sin_embed = self.sin_dropout(tf.sin(sinusoid), training=training)
        cos_embed = self.cos_dropout(tf.cos(sinusoid), training=training)
        pos_embed = tf.concat([sin_embed, cos_embed], axis=-1)
        pos = tf.range(0, seq_len, dtype=dtype)
        pooled_pos = pos
        position_embeds_list = []
        for block_index in range(0, self.num_blocks):
            if block_index == 0:
                position_embeds_pooling = None
            else:
                pooled_pos = self.stride_pool_pos(pos, block_index)
                stride = 2 ** (block_index - 1)
                rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                rel_pos = rel_pos + zero_offset
                position_embeds_pooling = tf.gather(pos_embed, rel_pos, axis=0)
            pos = pooled_pos
            stride = 2 ** block_index
            rel_pos = self.relative_pos(pos, stride)
            rel_pos = rel_pos + zero_offset
            position_embeds_no_pooling = tf.gather(pos_embed, rel_pos, axis=0)
            position_embeds_list.append([position_embeds_no_pooling,
                position_embeds_pooling])
        return position_embeds_list

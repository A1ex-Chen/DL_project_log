def post_attention(self, h, attn_vec, residual=True, training=False):
    """Post-attention processing."""
    attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)
    attn_out = self.dropout(attn_out, training=training)
    if residual:
        attn_out = attn_out + h
    output = self.layer_norm(attn_out)
    return output

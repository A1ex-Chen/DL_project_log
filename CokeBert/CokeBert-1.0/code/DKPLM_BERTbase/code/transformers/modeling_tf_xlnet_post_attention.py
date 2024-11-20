def post_attention(self, inputs, residual=True, training=False):
    """Post-attention processing."""
    h, attn_vec = inputs
    attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)
    attn_out = self.dropout(attn_out, training=training)
    if residual:
        attn_out = attn_out + h
    output = self.layer_norm(attn_out)
    return output

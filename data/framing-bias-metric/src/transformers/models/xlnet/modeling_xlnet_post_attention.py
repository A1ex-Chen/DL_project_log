def post_attention(self, h, attn_vec, residual=True):
    """Post-attention processing."""
    attn_out = torch.einsum('ibnd,hnd->ibh', attn_vec, self.o)
    attn_out = self.dropout(attn_out)
    if residual:
        attn_out = attn_out + h
    output = self.layer_norm(attn_out)
    return output

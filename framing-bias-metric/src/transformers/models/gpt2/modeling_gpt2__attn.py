def _attn(self, q, k, v, attention_mask=None, head_mask=None,
    output_attentions=False):
    w = torch.matmul(q, k)
    if self.scale:
        w = w / float(v.size(-1)) ** 0.5
    nd, ns = w.size(-2), w.size(-1)
    if not self.is_cross_attention:
        mask = self.bias[:, :, ns - nd:ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
    if attention_mask is not None:
        w = w + attention_mask
    w = nn.Softmax(dim=-1)(w)
    w = self.attn_dropout(w)
    if head_mask is not None:
        w = w * head_mask
    outputs = [torch.matmul(w, v)]
    if output_attentions:
        outputs.append(w)
    return outputs

def _attn(self, q, k, v, attention_mask=None, head_mask=None):
    w = torch.matmul(q, k)
    if self.scale:
        w = w / math.sqrt(v.size(-1))
    b = self.bias[:, :, :w.size(-2), :w.size(-1)]
    w = w * b + -10000.0 * (1 - b)
    if attention_mask is not None:
        w = w + attention_mask
    w = nn.Softmax(dim=-1)(w)
    w = self.attn_dropout(w)
    if head_mask is not None:
        w = w * head_mask
    outputs = [torch.matmul(w, v)]
    if self.output_attentions:
        outputs.append(w)
    return outputs

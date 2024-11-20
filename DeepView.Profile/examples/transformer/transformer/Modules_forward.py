def forward(self, q, k, v, mask=None):
    attn = torch.bmm(q, k.transpose(1, 2))
    attn = attn / self.temperature
    if mask is not None:
        attn = attn.masked_fill(mask, -np.inf)
    attn = self.softmax(attn)
    attn = self.dropout(attn)
    output = torch.bmm(attn, v)
    return output, attn

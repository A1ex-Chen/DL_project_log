def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).
        transpose(1, 2) for l, x in zip(self.linear_layers, (query, key,
        value))]
    x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout
        )
    x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    return self.output_linear(x)

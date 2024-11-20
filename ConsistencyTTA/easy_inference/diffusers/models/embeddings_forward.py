def forward(self, x):
    bs, length, width = x.size()

    def shape(x):
        x = x.view(bs, -1, self.num_heads, self.dim_per_head)
        x = x.transpose(1, 2)
        x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
        x = x.transpose(1, 2)
        return x
    class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x
        .dtype)
    x = torch.cat([class_token, x], dim=1)
    q = shape(self.q_proj(class_token))
    k = shape(self.k_proj(x))
    v = shape(self.v_proj(x))
    scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
    weight = torch.einsum('bct,bcs->bts', q * scale, k * scale)
    weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
    a = torch.einsum('bts,bcs->bct', weight, v)
    a = a.reshape(bs, -1, 1).transpose(1, 2)
    return a[:, 0, :]

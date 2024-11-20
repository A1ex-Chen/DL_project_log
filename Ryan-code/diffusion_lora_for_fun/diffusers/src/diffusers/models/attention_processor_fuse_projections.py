@torch.no_grad()
def fuse_projections(self, fuse=True):
    device = self.to_q.weight.data.device
    dtype = self.to_q.weight.data.dtype
    if not self.is_cross_attention:
        concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.
            weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]
        self.to_qkv = nn.Linear(in_features, out_features, bias=self.
            use_bias, device=device, dtype=dtype)
        self.to_qkv.weight.copy_(concatenated_weights)
        if self.use_bias:
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.
                bias.data, self.to_v.bias.data])
            self.to_qkv.bias.copy_(concatenated_bias)
    else:
        concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.
            weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]
        self.to_kv = nn.Linear(in_features, out_features, bias=self.
            use_bias, device=device, dtype=dtype)
        self.to_kv.weight.copy_(concatenated_weights)
        if self.use_bias:
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.
                bias.data])
            self.to_kv.bias.copy_(concatenated_bias)
    self.fused_projections = fuse

def forward(self, features, **kwargs):
    x = features[:, 0, :]
    x = self.dropout(x)
    x = self.dense(x)
    x = self.activation_fn(x)
    x = self.dropout(x)
    x = self.out_proj(x)
    return x

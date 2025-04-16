def forward(self, x):
    return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2
        ], x[..., 1::2, 1::2]], 1)

def forward(self, x, time_embed, context=None, context_mask=None,
    image_mask=None):
    height, width = x.shape[-2:]
    out = self.in_norm(x, time_embed)
    out = out.reshape(x.shape[0], -1, height * width).permute(0, 2, 1)
    context = context if context is not None else out
    if context_mask is not None:
        context_mask = context_mask.to(dtype=context.dtype)
    out = self.attention(out, context, context_mask)
    out = out.permute(0, 2, 1).unsqueeze(-1).reshape(out.shape[0], -1,
        height, width)
    x = x + out
    out = self.out_norm(x, time_embed)
    out = self.feed_forward(out)
    x = x + out
    return x

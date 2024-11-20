def bits_to_decimal(x, bits=BITS):
    """expects bits from -1 to 1, outputs image tensor from 0 to 1"""
    device = x.device
    x = (x > 0).int()
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device, dtype=torch.int32
        )
    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b (c d) h w -> b c d h w', d=8)
    dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
    return (dec / 255).clamp(0.0, 1.0)

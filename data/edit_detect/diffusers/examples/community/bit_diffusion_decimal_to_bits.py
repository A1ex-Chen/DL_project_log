def decimal_to_bits(x, bits=BITS):
    """expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1"""
    device = x.device
    x = (x * 255).int().clamp(0, 255)
    mask = 2 ** torch.arange(bits - 1, -1, -1, device=device)
    mask = rearrange(mask, 'd -> d 1 1')
    x = rearrange(x, 'b c h w -> b c 1 h w')
    bits = (x & mask != 0).float()
    bits = rearrange(bits, 'b c d h w -> b (c d) h w')
    bits = bits * 2 - 1
    return bits

def hook(module, input, output):
    x = input[0]
    B, N, C = x.shape
    qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.
        num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1) * module.scale
    attn = attn.softmax(dim=-1)
    attention[name] = attn

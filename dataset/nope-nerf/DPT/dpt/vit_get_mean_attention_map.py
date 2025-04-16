def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])
        ).float()
    attn = torch.nn.functional.interpolate(attn, size=shape[2:], mode=
        'bicubic', align_corners=False).squeeze(0)
    all_attn = torch.mean(attn, 0)
    return all_attn

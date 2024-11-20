def coerce_attn_resolutions(attn_resolutions):
    attn_resolutions = list(attn_resolutions)
    attn_resolutions_ = []
    for ar in attn_resolutions:
        if isinstance(ar, (list, tuple)):
            attn_resolutions_.append(list(ar))
        else:
            attn_resolutions_.append([ar, ar])
    return attn_resolutions_

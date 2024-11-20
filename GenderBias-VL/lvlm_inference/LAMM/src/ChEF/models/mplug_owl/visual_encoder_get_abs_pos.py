def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size != tgt_size:
        return F.interpolate(abs_pos.float().reshape(1, src_size, src_size,
            -1).permute(0, 3, 1, 2), size=(tgt_size, tgt_size), mode=
            'bicubic', align_corners=False).permute(0, 2, 3, 1).flatten(0, 2
            ).to(dtype=dtype)
    else:
        return abs_pos

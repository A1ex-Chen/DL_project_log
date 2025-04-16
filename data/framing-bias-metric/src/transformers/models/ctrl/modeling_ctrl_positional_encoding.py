def positional_encoding(position, d_model_size, dtype):
    angle_rads = angle_defn(torch.arange(position, dtype=dtype).unsqueeze(1
        ), torch.arange(d_model_size, dtype=dtype).unsqueeze(0), d_model_size)
    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])
    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

def quantize_cache_v(fdata, bits, qmax, qmin):
    qtype = torch.uint8
    device = fdata.device
    shape = fdata.shape
    fdata_cal = torch.flatten(fdata, 2)
    fmax = torch.amax(fdata_cal, dim=-1, keepdim=True)
    fmin = torch.amin(fdata_cal, dim=-1, keepdim=True)
    if qmax.device != fmax.device:
        qmax = qmax.to(device)
        qmin = qmin.to(device)
    scale = (fmax - fmin) / (qmax - qmin)
    zero = qmin - fmin / scale
    scale = scale.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    zero = zero.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    res_data = fdata / scale + zero
    qdata = torch.clamp(res_data, qmin, qmax).to(qtype)
    return qdata.contiguous(), scale, zero

def extract_scales(model):
    blocks = []
    extract_blocks_into_list(model['model'], blocks)
    scales = []
    for b in blocks:
        assert isinstance(b, LinearAddBlock)
        if hasattr(b, 'scale_identity'):
            scales.append((b.scale_identity.weight.detach(), b.scale_1x1.
                weight.detach(), b.scale_conv.weight.detach()))
        else:
            scales.append((b.scale_1x1.weight.detach(), b.scale_conv.weight
                .detach()))
        print('extract scales: ', scales[-1][-2].mean(), scales[-1][-1].mean())
    return scales

def _resize_with_antialiasing(input, size, interpolation='bicubic',
    align_corners=True):
    h, w = input.shape[-2:]
    factors = h / size[0], w / size[1]
    sigmas = max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) /
        2.0, 0.001)
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if ks[0] % 2 == 0:
        ks = ks[0] + 1, ks[1]
    if ks[1] % 2 == 0:
        ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(input, size=size, mode=
        interpolation, align_corners=align_corners)
    return output

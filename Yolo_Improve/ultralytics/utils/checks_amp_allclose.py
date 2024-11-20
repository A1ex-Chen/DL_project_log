def amp_allclose(m, im):
    """All close FP32 vs AMP results."""
    a = m(im, device=device, verbose=False)[0].boxes.data
    with torch.cuda.amp.autocast(True):
        b = m(im, device=device, verbose=False)[0].boxes.data
    del m
    return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)

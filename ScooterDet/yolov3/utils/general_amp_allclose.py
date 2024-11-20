def amp_allclose(model, im):
    m = AutoShape(model, verbose=False)
    a = m(im).xywhn[0]
    m.amp = True
    b = m(im).xywhn[0]
    return a.shape == b.shape and torch.allclose(a, b, atol=0.1)

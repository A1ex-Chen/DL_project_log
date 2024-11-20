def CLIP_preprocess(self, x):
    dtype = x.dtype
    if isinstance(x, torch.Tensor):
        if x.min() < -1.0 or x.max() > 1.0:
            raise ValueError(
                'Expected input tensor to have values in the range [-1, 1]')
    x = kornia.geometry.resize(x.to(torch.float32), (224, 224),
        interpolation='bicubic', align_corners=True, antialias=False).to(dtype
        =dtype)
    x = (x + 1.0) / 2.0
    x = kornia.enhance.normalize(x, torch.Tensor([0.48145466, 0.4578275, 
        0.40821073]), torch.Tensor([0.26862954, 0.26130258, 0.27577711]))
    return x

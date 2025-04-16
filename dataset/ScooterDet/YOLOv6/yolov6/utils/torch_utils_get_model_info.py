def get_model_info(model, img_size=640):
    """Get model Params and GFlops.
    Code base on https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/model_utils.py
    """
    from thop import profile
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters(
        )).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1000000.0
    flops /= 1000000000.0
    img_size = img_size if isinstance(img_size, list) else [img_size, img_size]
    flops *= img_size[0] * img_size[1] / stride / stride * 2
    info = 'Params: {:.2f}M, Gflops: {:.2f}'.format(params, flops)
    return info

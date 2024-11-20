def __init__(self, name: str='resnet50_fc512', path: str=model_dict[
    'resnet50_fc512_MSMT17'], device: str='cpu', image_size: tuple[int, int
    ]=(256, 128), pixel_mean: list[float]=field(default_factory=lambda : [
    0.485, 0.456, 0.406]), pixel_std: list[float]=field(default_factory=lambda
    : [0.229, 0.224, 0.225]), pixel_norm: bool=True, verbose: bool=False):
    super().__init__(name, path, device, image_size, pixel_mean, pixel_std,
        pixel_norm, verbose)

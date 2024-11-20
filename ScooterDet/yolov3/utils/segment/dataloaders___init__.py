def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=
    None, rect=False, image_weights=False, cache_images=False, single_cls=
    False, stride=32, pad=0, min_items=0, prefix='', downsample_ratio=1,
    overlap=False):
    super().__init__(path, img_size, batch_size, augment, hyp, rect,
        image_weights, cache_images, single_cls, stride, pad, min_items, prefix
        )
    self.downsample_ratio = downsample_ratio
    self.overlap = overlap

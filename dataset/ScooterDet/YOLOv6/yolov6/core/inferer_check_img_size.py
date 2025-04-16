def check_img_size(self, img_size, s=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):
        new_size = max(self.make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):
        new_size = [max(self.make_divisible(x, int(s)), floor) for x in
            img_size]
    else:
        raise Exception(f'Unsupported type of img_size: {type(img_size)}')
    if new_size != img_size:
        print(
            f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}'
            )
    return new_size if isinstance(img_size, list) else [new_size] * 2

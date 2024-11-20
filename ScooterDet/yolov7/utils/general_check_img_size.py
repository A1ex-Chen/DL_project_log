def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(
            'WARNING: --img-size %g must be multiple of max stride %g, updating to %g'
             % (img_size, s, new_size))
    return new_size

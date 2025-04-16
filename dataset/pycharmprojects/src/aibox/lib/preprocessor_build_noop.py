@staticmethod
def build_noop() ->'Preprocessor':
    return Preprocessor(image_resized_width=-1, image_resized_height=-1,
        image_min_side=-1, image_max_side=-1, image_side_divisor=1)

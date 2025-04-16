def extra_repr(self):
    return (
        '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'
        .format(**self.__dict__))

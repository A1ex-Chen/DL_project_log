@staticmethod
def _scale_depth(depth_coeff):

    def _sd(num_repeat):
        return int(math.ceil(num_repeat * depth_coeff))
    return _sd

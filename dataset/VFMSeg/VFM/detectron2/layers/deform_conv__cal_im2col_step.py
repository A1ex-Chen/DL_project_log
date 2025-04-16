@staticmethod
@lru_cache(maxsize=128)
def _cal_im2col_step(input_size, default_size):
    """
        Calculate proper im2col step size, which should be divisible by input_size and not larger
        than prefer_size. Meanwhile the step size should be as large as possible to be more
        efficient. So we choose the largest one among all divisors of input_size which are smaller
        than prefer_size.
        :param input_size: input batch size .
        :param default_size: default preferred im2col step size.
        :return: the largest proper step size.
        """
    if input_size <= default_size:
        return input_size
    best_step = 1
    for step in range(2, min(int(math.sqrt(input_size)) + 1, default_size)):
        if input_size % step == 0:
            if input_size // step <= default_size:
                return input_size // step
            best_step = step
    return best_step

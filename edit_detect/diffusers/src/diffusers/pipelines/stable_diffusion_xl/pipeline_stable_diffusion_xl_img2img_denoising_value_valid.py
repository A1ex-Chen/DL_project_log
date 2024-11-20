def denoising_value_valid(dnv):
    return isinstance(dnv, float) and 0 < dnv < 1

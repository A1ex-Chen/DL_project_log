@staticmethod
def _get_stddev_equivalent_to_msra_fill(kernel_size, fan_out):
    """Returns the stddev of random normal initialization as MSRAFill."""
    return (2 / (kernel_size[0] * kernel_size[1] * fan_out)) ** 0.5

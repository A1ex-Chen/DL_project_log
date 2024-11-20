@staticmethod
def _scale_width(width_coeff, divisor=8):

    def _sw(num_channels):
        num_channels *= width_coeff
        rounded_num_channels = max(divisor, int(num_channels + divisor / 2) //
            divisor * divisor)
        if rounded_num_channels < 0.9 * num_channels:
            rounded_num_channels += divisor
        return rounded_num_channels
    return _sw

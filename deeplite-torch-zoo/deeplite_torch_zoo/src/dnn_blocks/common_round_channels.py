def round_channels(channels, divisor=8):
    rounded_channels = max(int(channels + divisor / 2.0) // divisor *
        divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels

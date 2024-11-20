def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0
        )
    window = Variable(_2D_window.expand(channel, 1, window_size,
        window_size).contiguous())
    return window

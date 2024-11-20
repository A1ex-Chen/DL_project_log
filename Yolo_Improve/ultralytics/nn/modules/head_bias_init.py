def bias_init(self):
    """Initialize Detect() biases, WARNING: requires stride availability."""
    m = self
    for a, b, s in zip(m.cv2, m.cv3, m.stride):
        a[-1].bias.data[:] = 1.0

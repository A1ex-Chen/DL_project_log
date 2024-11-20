def conv_bytes_flops(self, N, C, H, W, K, P, Q, R, S, g, t):
    f = 2 * N * K * P * Q * C * R * S / g
    elems = N * C * H * W + K * C * R * S / g + N * K * P * Q
    b = elems * Utility.typeToBytes(t)
    return b, f

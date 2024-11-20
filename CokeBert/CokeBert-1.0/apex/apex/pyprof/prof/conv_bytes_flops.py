def bytes_flops(self):
    N, C, H, W, K, P, Q, R, S, ph, pw, U, V, dh, dw, g, t = self.params(
        ).values()
    if any(x in self.name for x in Conv.convAuxList + Conv.winoAuxList +
        Conv.fftAuxList + Conv.miscAuxList):
        bytes, flops = [0, 0]
    elif any(x in self.name for x in Conv.convList + Conv.winoList + Conv.
        fftList + Conv.miscList):
        if g == 1:
            bytes, flops = self.conv_bytes_flops(N, C, H, W, K, P, Q, R, S,
                g, t)
        elif '2d_grouped_direct_kernel' in self.name:
            bytes, flops = self.conv_bytes_flops(N, C, H, W, K, P, Q, R, S,
                g, t)
        elif 'spatialDepthwiseConvolutionUpdateOutput' in self.name:
            bytes, flops = self.conv_bytes_flops(N, C, H, W, K, P, Q, R, S,
                g, t)
        else:
            bytes, flops = self.conv_bytes_flops(N, C / g, H, W, K / g, P,
                Q, R, S, 1, t)
    elif 'calc_bias_diff' in self.name:
        elems = N * K * P * Q
        flops = elems
        bytes = 2 * elems * Utility.typeToBytes(t)
    else:
        bytes, flops = [0, 0]
    return bytes, flops

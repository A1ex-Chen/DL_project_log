def encode(self, x):
    x = self.freq_split_subband(x)
    h = self.encoder(x)
    moments = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(moments)
    return posterior

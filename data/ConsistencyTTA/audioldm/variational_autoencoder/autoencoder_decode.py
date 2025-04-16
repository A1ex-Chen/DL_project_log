def decode(self, z, use_ema=False):
    if use_ema and (not hasattr(self, 'ema_decoder') or self.ema_decoder is
        None):
        print(
            'VAE does not have EMA modules, but specified use_ema. Using the none-EMA modules instead.'
            )
    if use_ema and hasattr(self, 'ema_decoder'
        ) and self.ema_decoder is not None:
        z = self.ema_post_quant_conv(z)
        dec = self.ema_decoder(z)
    else:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
    return self.freq_merge_subband(dec)

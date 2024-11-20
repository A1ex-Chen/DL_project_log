def decode_to_waveform(self, dec, allow_grad=False):
    dec = dec.squeeze(1).permute(0, 2, 1)
    wav_reconstruction = vocoder_infer(dec, self.vocoder, allow_grad=allow_grad
        )
    return wav_reconstruction

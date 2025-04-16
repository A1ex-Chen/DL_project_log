def freq_split_subband(self, fbank):
    if self.subband == 1 or self.image_key != 'stft':
        return fbank
    bs, ch, tstep, fbins = fbank.size()
    assert fbank.size(-1) % self.subband == 0
    assert ch == 1
    return fbank.squeeze(1).reshape(bs, tstep, self.subband, fbins // self.
        subband).permute(0, 2, 1, 3)

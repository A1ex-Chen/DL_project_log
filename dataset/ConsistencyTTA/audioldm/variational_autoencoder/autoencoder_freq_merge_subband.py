def freq_merge_subband(self, subband_fbank):
    if self.subband == 1 or self.image_key != 'stft':
        return subband_fbank
    assert subband_fbank.size(1) == self.subband
    bs, sub_ch, tstep, fbins = subband_fbank.size()
    return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1
        )

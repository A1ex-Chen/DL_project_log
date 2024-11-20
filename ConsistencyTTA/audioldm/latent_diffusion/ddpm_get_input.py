def get_input(self, batch, k):
    fbank, log_magnitudes_stft, label_indices, fname, waveform, text = batch
    ret = {}
    ret['fbank'] = fbank.unsqueeze(1).to(memory_format=torch.contiguous_format
        ).float()
    ret['stft'] = log_magnitudes_stft.to(memory_format=torch.contiguous_format
        ).float()
    ret['waveform'] = waveform.to(memory_format=torch.contiguous_format).float(
        )
    ret['text'] = list(text)
    ret['fname'] = fname
    return ret[k]

def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length)
        waveform = torch.cat([waveform, pad_wav.to(waveform.device)])
        return waveform

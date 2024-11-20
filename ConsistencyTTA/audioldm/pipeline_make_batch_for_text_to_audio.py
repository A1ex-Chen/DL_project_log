def make_batch_for_text_to_audio(text, waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print('Warning: Batchsize must be at least 1. Batchsize is set to .')
    if fbank is None:
        fbank = torch.zeros((batchsize, 1024, 64))
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize
    stft = torch.zeros((batchsize, 1024, 512))
    if waveform is None:
        waveform = torch.zeros((batchsize, 160000))
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
    fname = [''] * batchsize
    batch = fbank, stft, None, fname, waveform, text
    return batch

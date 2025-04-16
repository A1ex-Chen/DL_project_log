def mel_spectrogram_to_waveform(self, mel_spectrogram):
    if mel_spectrogram.dim() == 4:
        mel_spectrogram = mel_spectrogram.squeeze(1)
    waveform = self.vocoder(mel_spectrogram)
    waveform = waveform.cpu()
    return waveform

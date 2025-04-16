def mel_spectrogram_to_waveform(self, mel):
    if len(mel.size()) == 4:
        mel = mel.squeeze(1)
    mel = mel.permute(0, 2, 1)
    waveform = self.first_stage_model.vocoder(mel)
    waveform = waveform.cpu().detach().numpy()
    return waveform

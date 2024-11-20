def get_mel(audio_data, audio_cfg):
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=audio_cfg[
        'sample_rate'], n_fft=audio_cfg['window_size'], win_length=
        audio_cfg['window_size'], hop_length=audio_cfg['hop_size'], center=
        True, pad_mode='reflect', power=2.0, norm=None, onesided=True,
        n_mels=64, f_min=audio_cfg['fmin'], f_max=audio_cfg['fmax']).to(
        audio_data.device)
    mel = mel(audio_data)
    mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)
    return mel.T

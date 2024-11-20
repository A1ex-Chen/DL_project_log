def __init__(self, sampling_rate, device, backbone='cnn14') ->None:
    self.device = device
    self.backbone = backbone
    self.sampling_rate = sampling_rate
    self.frechet = FrechetAudioDistance(use_pca=False, use_activation=False)
    self.lsd_metric = AudioMetrics(self.sampling_rate)
    self.frechet.model = self.frechet.model.to(self.device)
    features_list = ['2048', 'logits']
    if self.sampling_rate == 16000:
        self.mel_model = Cnn14(features_list=features_list, sample_rate=
            16000, window_size=512, hop_size=160, mel_bins=64, fmin=50,
            fmax=8000, classes_num=527).to(self.device)
    elif self.sampling_rate == 32000:
        self.mel_model = Cnn14(features_list=features_list, sample_rate=
            32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50,
            fmax=14000, classes_num=527).to(self.device)
    else:
        raise ValueError(
            'We only support the evaluation on 16kHz and 32kHz sampling rates.'
            )
    self.mel_model.eval()
    self.fbin_mean, self.fbin_std = None, None
    if self.sampling_rate == 16000:
        self._stft = Audio.TacotronSTFT(filter_length=512, hop_length=160,
            win_length=512, n_mel_channels=64, sampling_rate=16000,
            mel_fmin=50, mel_fmax=8000)
    elif self.sampling_rate == 32000:
        self._stft = Audio.TacotronSTFT(filter_length=1024, hop_length=320,
            win_length=1024, n_mel_channels=64, sampling_rate=32000,
            mel_fmin=50, mel_fmax=14000)
    else:
        raise ValueError(
            'We only support the evaluation on 16kHz and 32kHz sampling rates.'
            )
    self.clap_model = CLAP_Module(enable_fusion=False, amodel='HTSAT-base').to(
        device)
    self.clap_model.load_ckpt('ckpt/music_audioset_epoch_15_esc_90.14.pt',
        verbose=False)

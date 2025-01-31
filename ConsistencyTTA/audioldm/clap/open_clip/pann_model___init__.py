def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
    classes_num, enable_fusion=False, fusion_type='None'):
    super(Cnn10, self).__init__()
    window = 'hann'
    center = True
    pad_mode = 'reflect'
    ref = 1.0
    amin = 1e-10
    top_db = None
    self.enable_fusion = enable_fusion
    self.fusion_type = fusion_type
    self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=
        hop_size, win_length=window_size, window=window, center=center,
        pad_mode=pad_mode, freeze_parameters=True)
    self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=
        window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=
        amin, top_db=top_db, freeze_parameters=True)
    self.spec_augmenter = SpecAugmentation(time_drop_width=64,
        time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
    self.bn0 = nn.BatchNorm2d(64)
    self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
    self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
    self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
    self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
    self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
    self.fc1 = nn.Linear(1024, 1024, bias=True)
    self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
    self.init_weight()

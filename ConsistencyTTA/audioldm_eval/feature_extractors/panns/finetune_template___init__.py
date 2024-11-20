def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
    classes_num, freeze_base):
    """Classifier for a new task using pretrained Cnn14 as a sub module."""
    super(Transfer_Cnn14, self).__init__()
    audioset_classes_num = 527
    self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, audioset_classes_num)
    self.fc_transfer = nn.Linear(2048, classes_num, bias=True)
    if freeze_base:
        for param in self.base.parameters():
            param.requires_grad = False
    self.init_weights()

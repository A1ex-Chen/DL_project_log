def __init__(self, ddconfig=None, lossconfig=None, image_key='fbank',
    embed_dim=None, time_shuffle=1, subband=1, ckpt_path=None,
    reload_from_ckpt=None, ignore_keys=[], colorize_nlabels=None, monitor=
    None, base_learning_rate=1e-05, scale_factor=1):
    super().__init__()
    self.encoder = Encoder(**ddconfig)
    self.decoder = Decoder(**ddconfig)
    self.ema_decoder = None
    self.subband = int(subband)
    if self.subband > 1:
        print('Use subband decomposition %s' % self.subband)
    self.quant_conv = nn.Conv2d(2 * ddconfig['z_channels'], 2 * embed_dim, 1)
    self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)
    self.ema_post_quant_conv = None
    self.vocoder = get_vocoder(None, 'cpu')
    self.embed_dim = embed_dim
    if monitor is not None:
        self.monitor = monitor
    self.time_shuffle = time_shuffle
    self.reload_from_ckpt = reload_from_ckpt
    self.reloaded = False
    self.mean, self.std = None, None
    self.scale_factor = scale_factor

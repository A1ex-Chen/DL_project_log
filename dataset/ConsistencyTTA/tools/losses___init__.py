def __init__(self, vae: Module, reduction: str='instance', mse_weight:
    float=1.0, clap_weight: float=1.0):
    super().__init__()
    self.vae = vae
    self.reduction = reduction
    self.sr = 16000
    self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base'
        )
    self.clap.load_ckpt('ckpt/music_audioset_epoch_15_esc_90.14.pt')
    self.clap.eval()
    self.clap.requires_grad_(False)
    self.mse_weight = mse_weight
    self.clap_weight = clap_weight

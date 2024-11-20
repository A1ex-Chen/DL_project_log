@classmethod
def from_config(cls, cfg=None):
    if cfg is None:
        cfg = OmegaConf.create()
    visual_ft = cfg.get('visual_ft', ['i3d_rgb'])
    audio_ft = cfg.get('audio_ft', ['vggish'])
    return cls(visual_ft=visual_ft, audio_ft=audio_ft)

def get_vocoder(config, device):
    config = hifigan.AttrDict(HIFIGAN_16K_64)
    vocoder = hifigan.Generator(config)
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder

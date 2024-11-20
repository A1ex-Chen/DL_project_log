def convert_hifigan_checkpoint(checkpoint, config):
    """
    Takes a state dict and config, and returns a converted HiFiGAN vocoder checkpoint.
    """
    vocoder_state_dict = extract_sub_model(checkpoint, key_prefix=
        'first_stage_model.vocoder.')
    for i in range(len(config.upsample_rates)):
        vocoder_state_dict[f'upsampler.{i}.weight'] = vocoder_state_dict.pop(
            f'ups.{i}.weight')
        vocoder_state_dict[f'upsampler.{i}.bias'] = vocoder_state_dict.pop(
            f'ups.{i}.bias')
    if not config.normalize_before:
        vocoder_state_dict['mean'] = torch.zeros(config.model_in_dim)
        vocoder_state_dict['scale'] = torch.ones(config.model_in_dim)
    return vocoder_state_dict

def create_transformers_vocoder_config(original_config):
    """
    Creates a config for transformers SpeechT5HifiGan based on the config of the vocoder model.
    """
    vocoder_params = original_config['model']['params']['vocoder_config'][
        'params']
    config = {'model_in_dim': vocoder_params['num_mels'], 'sampling_rate':
        vocoder_params['sampling_rate'], 'upsample_initial_channel':
        vocoder_params['upsample_initial_channel'], 'upsample_rates': list(
        vocoder_params['upsample_rates']), 'upsample_kernel_sizes': list(
        vocoder_params['upsample_kernel_sizes']), 'resblock_kernel_sizes':
        list(vocoder_params['resblock_kernel_sizes']),
        'resblock_dilation_sizes': [list(resblock_dilation) for
        resblock_dilation in vocoder_params['resblock_dilation_sizes']],
        'normalize_before': False}
    return config

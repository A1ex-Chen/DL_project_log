def get_asymmetric_autoencoder_kl_from_original_checkpoint(scale: Literal[
    '1.5', '2'], original_checkpoint_path: str, map_location: torch.device
    ) ->AsymmetricAutoencoderKL:
    print('Loading original state_dict')
    original_state_dict = torch.load(original_checkpoint_path, map_location
        =map_location)
    original_state_dict = original_state_dict['state_dict']
    print('Converting state_dict')
    converted_state_dict = convert_asymmetric_autoencoder_kl_state_dict(
        original_state_dict)
    kwargs = (ASYMMETRIC_AUTOENCODER_KL_x_1_5_CONFIG if scale == '1.5' else
        ASYMMETRIC_AUTOENCODER_KL_x_2_CONFIG)
    print('Initializing AsymmetricAutoencoderKL model')
    asymmetric_autoencoder_kl = AsymmetricAutoencoderKL(**kwargs)
    print('Loading weight from converted state_dict')
    asymmetric_autoencoder_kl.load_state_dict(converted_state_dict)
    asymmetric_autoencoder_kl.eval()
    print('AsymmetricAutoencoderKL successfully initialized')
    return asymmetric_autoencoder_kl

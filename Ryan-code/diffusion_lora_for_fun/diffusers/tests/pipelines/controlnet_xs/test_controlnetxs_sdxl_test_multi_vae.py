def test_multi_vae(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    block_out_channels = pipe.vae.config.block_out_channels
    norm_num_groups = pipe.vae.config.norm_num_groups
    vae_classes = [AutoencoderKL, AsymmetricAutoencoderKL,
        ConsistencyDecoderVAE, AutoencoderTiny]
    configs = [get_autoencoder_kl_config(block_out_channels,
        norm_num_groups), get_asym_autoencoder_kl_config(block_out_channels,
        norm_num_groups), get_consistency_vae_config(block_out_channels,
        norm_num_groups), get_autoencoder_tiny_config(block_out_channels)]
    out_np = pipe(**self.get_dummy_inputs_by_type(torch_device,
        input_image_type='np'))[0]
    for vae_cls, config in zip(vae_classes, configs):
        vae = vae_cls(**config)
        vae = vae.to(torch_device)
        components['vae'] = vae
        vae_pipe = self.pipeline_class(**components)
        vae_pipe.to(torch_device)
        vae_pipe.set_progress_bar_config(disable=None)
        out_vae_np = vae_pipe(**self.get_dummy_inputs_by_type(torch_device,
            input_image_type='np'))[0]
        assert out_vae_np.shape == out_np.shape

def test_freeu_disabled(self):
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'hey'
    output = sd_pipe(prompt, num_inference_steps=1, output_type='np',
        generator=torch.manual_seed(0)).images
    sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    sd_pipe.disable_freeu()
    freeu_keys = {'s1', 's2', 'b1', 'b2'}
    for upsample_block in sd_pipe.unet.up_blocks:
        for key in freeu_keys:
            assert getattr(upsample_block, key
                ) is None, f'Disabling of FreeU should have set {key} to None.'
    output_no_freeu = sd_pipe(prompt, num_inference_steps=1, output_type=
        'np', generator=torch.manual_seed(0)).images
    assert np.allclose(output[0, -3:, -3:, -1], output_no_freeu[0, -3:, -3:,
        -1]
        ), 'Disabling of FreeU should lead to results similar to the default pipeline results.'

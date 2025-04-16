@skip_mps
def test_freeu_enabled(self):
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'hey'
    output = sd_pipe(prompt, num_inference_steps=1, output_type='np',
        generator=torch.manual_seed(0)).images
    sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    output_freeu = sd_pipe(prompt, num_inference_steps=1, output_type='np',
        generator=torch.manual_seed(0)).images
    assert not np.allclose(output[0, -3:, -3:, -1], output_freeu[0, -3:, -3
        :, -1]), 'Enabling of FreeU should lead to different results.'

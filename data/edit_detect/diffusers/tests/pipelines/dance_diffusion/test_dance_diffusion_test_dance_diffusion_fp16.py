def test_dance_diffusion_fp16(self):
    device = torch_device
    pipe = DanceDiffusionPipeline.from_pretrained('harmonai/maestro-150k',
        torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    output = pipe(generator=generator, num_inference_steps=100,
        audio_length_in_s=4.096)
    audio = output.audios
    audio_slice = audio[0, -3:, -3:]
    assert audio.shape == (1, 2, pipe.unet.config.sample_size)
    expected_slice = np.array([-0.0367, -0.0488, -0.0771, -0.0525, -0.0444,
        -0.0341])
    assert np.abs(audio_slice.flatten() - expected_slice).max() < 0.01

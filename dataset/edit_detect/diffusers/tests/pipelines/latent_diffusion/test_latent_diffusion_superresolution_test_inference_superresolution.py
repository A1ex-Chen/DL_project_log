def test_inference_superresolution(self):
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/vq_diffusion/teddy_bear_pool.png'
        )
    init_image = init_image.resize((64, 64), resample=PIL_INTERPOLATION[
        'lanczos'])
    ldm = LDMSuperResolutionPipeline.from_pretrained(
        'duongna/ldm-super-resolution')
    ldm.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = ldm(image=init_image, generator=generator, num_inference_steps=
        20, output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.7644, 0.7679, 0.7642, 0.7633, 0.7666, 
        0.756, 0.7425, 0.7257, 0.6907])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

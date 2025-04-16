def test_unclip_karlo_cpu_fp32(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unclip/karlo_v1_alpha_horse_cpu.npy'
        )
    pipeline = UnCLIPPipeline.from_pretrained('kakaobrain/karlo-v1-alpha')
    pipeline.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    output = pipeline('horse', num_images_per_prompt=1, generator=generator,
        output_type='np')
    image = output.images[0]
    assert image.shape == (256, 256, 3)
    assert np.abs(expected_image - image).max() < 0.1

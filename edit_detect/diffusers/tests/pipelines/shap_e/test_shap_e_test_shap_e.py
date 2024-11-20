def test_shap_e(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/test_shap_e_np_out.npy'
        )
    pipe = ShapEPipeline.from_pretrained('openai/shap-e')
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device=torch_device).manual_seed(0)
    images = pipe('a shark', generator=generator, guidance_scale=15.0,
        num_inference_steps=64, frame_size=64, output_type='np').images[0]
    assert images.shape == (20, 64, 64, 3)
    assert_mean_pixel_difference(images, expected_image)

def test_stable_unclip(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_l_anime_turtle_fp16.npy'
        )
    pipe = StableUnCLIPPipeline.from_pretrained('fusing/stable-unclip-2-1-l',
        torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipe('anime turle', generator=generator, output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    assert_mean_pixel_difference(image, expected_image)

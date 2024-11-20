def test_stable_unclip_h_img2img(self):
    input_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/turtle.png'
        )
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/stable_unclip_2_1_h_img2img_anime_turtle_fp16.npy'
        )
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        'fusing/stable-unclip-2-1-h-img2img', torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipe(input_image, 'anime turle', generator=generator,
        output_type='np')
    image = output.images[0]
    assert image.shape == (768, 768, 3)
    assert_mean_pixel_difference(image, expected_image)

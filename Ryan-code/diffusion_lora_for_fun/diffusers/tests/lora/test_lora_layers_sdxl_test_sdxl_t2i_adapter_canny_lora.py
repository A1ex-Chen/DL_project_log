def test_sdxl_t2i_adapter_canny_lora(self):
    adapter = T2IAdapter.from_pretrained(
        'TencentARC/t2i-adapter-lineart-sdxl-1.0', torch_dtype=torch.float16
        ).to('cpu')
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', adapter=adapter,
        torch_dtype=torch.float16, variant='fp16')
    pipe.load_lora_weights('CiroN2022/toy-face', weight_name=
        'toy_face_sdxl.safetensors')
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device='cpu').manual_seed(0)
    prompt = 'toy'
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png'
        )
    images = pipe(prompt, image=image, generator=generator, output_type=
        'np', num_inference_steps=3).images
    assert images[0].shape == (768, 512, 3)
    image_slice = images[0, -3:, -3:, -1].flatten()
    expected_slice = np.array([0.4284, 0.4337, 0.4319, 0.4255, 0.4329, 
        0.428, 0.4338, 0.442, 0.4226])
    assert numpy_cosine_similarity_distance(image_slice, expected_slice
        ) < 0.0001

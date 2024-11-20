def test_stable_diffusion_adapter_sketch_sd14(self):
    adapter_model = 'TencentARC/t2iadapter_sketch_sd14v1'
    sd_model = 'CompVis/stable-diffusion-v1-4'
    prompt = 'cat'
    image_url = (
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/edge.png'
        )
    input_channels = 1
    out_url = (
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_sketch_sd14v1.npy'
        )
    image = load_image(image_url)
    expected_out = load_numpy(out_url)
    if input_channels == 1:
        image = image.convert('L')
    adapter = T2IAdapter.from_pretrained(adapter_model, torch_dtype=torch.
        float16)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model, adapter
        =adapter, safety_checker=None)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    generator = torch.Generator(device='cpu').manual_seed(0)
    out = pipe(prompt=prompt, image=image, generator=generator,
        num_inference_steps=2, output_type='np').images
    max_diff = numpy_cosine_similarity_distance(out.flatten(), expected_out
        .flatten())
    assert max_diff < 0.01

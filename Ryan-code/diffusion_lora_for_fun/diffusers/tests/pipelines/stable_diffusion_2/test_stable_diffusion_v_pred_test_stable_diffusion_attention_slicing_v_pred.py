def test_stable_diffusion_attention_slicing_v_pred(self):
    torch.cuda.reset_peak_memory_stats()
    model_id = 'stabilityai/stable-diffusion-2'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=
        torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'a photograph of an astronaut riding a horse'
    pipe.enable_attention_slicing()
    generator = torch.manual_seed(0)
    output_chunked = pipe([prompt], generator=generator, guidance_scale=7.5,
        num_inference_steps=10, output_type='np')
    image_chunked = output_chunked.images
    mem_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    assert mem_bytes < 5.5 * 10 ** 9
    pipe.disable_attention_slicing()
    generator = torch.manual_seed(0)
    output = pipe([prompt], generator=generator, guidance_scale=7.5,
        num_inference_steps=10, output_type='np')
    image = output.images
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes > 3 * 10 ** 9
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        image_chunked.flatten())
    assert max_diff < 0.001

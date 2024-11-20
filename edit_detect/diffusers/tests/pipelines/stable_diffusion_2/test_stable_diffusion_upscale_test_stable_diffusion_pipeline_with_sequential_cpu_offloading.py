def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png'
        )
    model_id = 'stabilityai/stable-diffusion-x4-upscaler'
    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id,
        torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    prompt = 'a cat sitting on a park bench'
    generator = torch.manual_seed(0)
    _ = pipe(prompt=prompt, image=image, generator=generator,
        num_inference_steps=5, output_type='np')
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 2.9 * 10 ** 9

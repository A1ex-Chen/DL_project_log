def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    init_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/init_image.png'
        )
    mask_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-inpaint/mask.png'
        )
    model_id = 'stabilityai/stable-diffusion-2-inpainting'
    pndm = PNDMScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id,
        safety_checker=None, scheduler=pndm, torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    prompt = 'Face of a yellow cat, high resolution, sitting on a park bench'
    generator = torch.manual_seed(0)
    _ = pipe(prompt=prompt, image=init_image, mask_image=mask_image,
        generator=generator, num_inference_steps=2, output_type='np')
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 2.65 * 10 ** 9

def test_sequential_cpu_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    controlnet = ControlNetModel.from_pretrained('lllyasviel/sd-controlnet-seg'
        )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, controlnet=
        controlnet)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    pipe.enable_sequential_cpu_offload()
    prompt = 'house'
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png'
        )
    _ = pipe(prompt, image, num_inference_steps=2, output_type='np')
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 4 * 10 ** 9

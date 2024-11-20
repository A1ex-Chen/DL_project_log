def test_stable_diffusion_adapter_pipeline_with_sequential_cpu_offloading(self
    ):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    adapter = T2IAdapter.from_pretrained('TencentARC/t2iadapter_seg_sd14v1')
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', adapter=adapter, safety_checker=None)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/motor.png'
        )
    pipe(prompt='foo', image=image, num_inference_steps=2)
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 5 * 10 ** 9

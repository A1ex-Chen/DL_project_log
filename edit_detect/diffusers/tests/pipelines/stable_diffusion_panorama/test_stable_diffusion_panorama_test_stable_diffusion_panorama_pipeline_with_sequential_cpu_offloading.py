def test_stable_diffusion_panorama_pipeline_with_sequential_cpu_offloading(self
    ):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    model_ckpt = 'stabilityai/stable-diffusion-2-base'
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder='scheduler'
        )
    pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt,
        scheduler=scheduler, safety_checker=None)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    inputs = self.get_inputs()
    _ = pipe(**inputs)
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 5.5 * 10 ** 9

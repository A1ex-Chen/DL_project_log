def test_stable_diffusion_pipeline_with_model_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None, torch_dtype=
        torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe(**inputs)
    mem_bytes = torch.cuda.max_memory_allocated()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None, torch_dtype=
        torch.float16)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    _ = pipe(**inputs)
    mem_bytes_offloaded = torch.cuda.max_memory_allocated()
    assert mem_bytes_offloaded < mem_bytes
    for module in (pipe.text_encoder, pipe.unet, pipe.vae):
        assert module.device == torch.device('cpu')

def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        'timbrooks/instruct-pix2pix', safety_checker=None, torch_dtype=
        torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    inputs = self.get_inputs()
    _ = pipe(**inputs)
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 2.2 * 10 ** 9

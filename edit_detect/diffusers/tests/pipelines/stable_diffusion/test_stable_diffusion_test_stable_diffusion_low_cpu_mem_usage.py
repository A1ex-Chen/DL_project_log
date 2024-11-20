def test_stable_diffusion_low_cpu_mem_usage(self):
    pipeline_id = 'CompVis/stable-diffusion-v1-4'
    start_time = time.time()
    pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(
        pipeline_id, torch_dtype=torch.float16)
    pipeline_low_cpu_mem_usage.to(torch_device)
    low_cpu_mem_usage_time = time.time() - start_time
    start_time = time.time()
    _ = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=
        torch.float16, low_cpu_mem_usage=False)
    normal_load_time = time.time() - start_time
    assert 2 * low_cpu_mem_usage_time < normal_load_time

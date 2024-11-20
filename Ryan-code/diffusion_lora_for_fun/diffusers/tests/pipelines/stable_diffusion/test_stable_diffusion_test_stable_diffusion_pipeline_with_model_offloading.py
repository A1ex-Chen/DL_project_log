def test_stable_diffusion_pipeline_with_model_offloading(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipe.unet.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    outputs = pipe(**inputs)
    mem_bytes = torch.cuda.max_memory_allocated()
    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', torch_dtype=torch.float16)
    pipe.unet.set_default_attn_processor()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    outputs_offloaded = pipe(**inputs)
    mem_bytes_offloaded = torch.cuda.max_memory_allocated()
    images = outputs.images
    offloaded_images = outputs_offloaded.images
    max_diff = numpy_cosine_similarity_distance(images.flatten(),
        offloaded_images.flatten())
    assert max_diff < 0.001
    assert mem_bytes_offloaded < mem_bytes
    assert mem_bytes_offloaded < 3.5 * 10 ** 9
    for module in (pipe.text_encoder, pipe.unet, pipe.vae):
        assert module.device == torch.device('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pipe.enable_attention_slicing()
    _ = pipe(**inputs)
    mem_bytes_slicing = torch.cuda.max_memory_allocated()
    assert mem_bytes_slicing < mem_bytes_offloaded
    assert mem_bytes_slicing < 3 * 10 ** 9

@require_torch_gpu
def test_stable_diffusion_attention_slicing(self):
    torch.cuda.reset_peak_memory_stats()
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-base', torch_dtype=torch.float16)
    pipe.unet.set_default_attn_processor()
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    image_sliced = pipe(**inputs).images
    mem_bytes = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    assert mem_bytes < 3.3 * 10 ** 9
    pipe.disable_attention_slicing()
    pipe.unet.set_default_attn_processor()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    image = pipe(**inputs).images
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes > 3.3 * 10 ** 9
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        image_sliced.flatten())
    assert max_diff < 0.005

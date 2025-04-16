def test_single_file_format_inference_is_same_as_pretrained(self):
    adapter = T2IAdapter.from_pretrained(
        'TencentARC/t2i-adapter-lineart-sdxl-1.0', torch_dtype=torch.float16)
    pipe_single_file = StableDiffusionXLAdapterPipeline.from_single_file(self
        .ckpt_path, adapter=adapter, torch_dtype=torch.float16,
        safety_checker=None)
    pipe_single_file.enable_model_cpu_offload()
    pipe_single_file.set_progress_bar_config(disable=None)
    inputs = self.get_inputs()
    images_single_file = pipe_single_file(**inputs).images[0]
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(self.repo_id,
        adapter=adapter, torch_dtype=torch.float16, safety_checker=None)
    pipe.enable_model_cpu_offload()
    inputs = self.get_inputs()
    images = pipe(**inputs).images[0]
    assert images_single_file.shape == (768, 512, 3)
    assert images.shape == (768, 512, 3)
    max_diff = numpy_cosine_similarity_distance(images.flatten(),
        images_single_file.flatten())
    assert max_diff < 0.005

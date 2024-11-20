def test_single_file_format_inference_is_same_as_pretrained(self):
    controlnet = ControlNetModel.from_pretrained(
        'diffusers/controlnet-depth-sdxl-1.0', torch_dtype=torch.float16)
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        controlnet=controlnet, torch_dtype=torch.float16)
    pipe_single_file.unet.set_default_attn_processor()
    pipe_single_file.enable_model_cpu_offload()
    pipe_single_file.set_progress_bar_config(disable=None)
    inputs = self.get_inputs(torch_device)
    single_file_images = pipe_single_file(**inputs).images[0]
    pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=
        controlnet, torch_dtype=torch.float16)
    pipe.unet.set_default_attn_processor()
    pipe.enable_model_cpu_offload()
    inputs = self.get_inputs(torch_device)
    images = pipe(**inputs).images[0]
    assert images.shape == (512, 512, 3)
    assert single_file_images.shape == (512, 512, 3)
    max_diff = numpy_cosine_similarity_distance(images[0].flatten(),
        single_file_images[0].flatten())
    assert max_diff < 0.05

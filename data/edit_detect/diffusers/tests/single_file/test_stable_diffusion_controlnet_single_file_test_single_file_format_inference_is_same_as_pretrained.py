def test_single_file_format_inference_is_same_as_pretrained(self):
    controlnet = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11p_sd15_canny')
    pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=
        controlnet)
    pipe.unet.set_default_attn_processor()
    pipe.enable_model_cpu_offload()
    pipe_sf = self.pipeline_class.from_single_file(self.ckpt_path,
        controlnet=controlnet)
    pipe_sf.unet.set_default_attn_processor()
    pipe_sf.enable_model_cpu_offload()
    inputs = self.get_inputs()
    output = pipe(**inputs).images[0]
    inputs = self.get_inputs()
    output_sf = pipe_sf(**inputs).images[0]
    max_diff = numpy_cosine_similarity_distance(output_sf.flatten(), output
        .flatten())
    assert max_diff < 0.001

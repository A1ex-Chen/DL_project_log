def test_single_file_format_inference_is_same_as_pretrained(self,
    expected_max_diff=0.0001):
    sf_pipe = self.pipeline_class.from_single_file(self.ckpt_path,
        torch_dtype=torch.float16, safety_checker=None)
    sf_pipe.unet.set_default_attn_processor()
    sf_pipe.enable_model_cpu_offload()
    inputs = self.get_inputs(torch_device)
    image_single_file = sf_pipe(**inputs).images[0]
    pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=
        torch.float16, safety_checker=None)
    pipe.unet.set_default_attn_processor()
    pipe.enable_model_cpu_offload()
    inputs = self.get_inputs(torch_device)
    image = pipe(**inputs).images[0]
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        image_single_file.flatten())
    assert max_diff < expected_max_diff

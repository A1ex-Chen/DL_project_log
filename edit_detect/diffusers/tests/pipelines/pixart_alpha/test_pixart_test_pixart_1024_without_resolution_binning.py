def test_pixart_1024_without_resolution_binning(self):
    generator = torch.manual_seed(0)
    pipe = PixArtAlphaPipeline.from_pretrained(self.ckpt_id_1024,
        torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    prompt = self.prompt
    height, width = 1024, 768
    num_inference_steps = 2
    image = pipe(prompt, height=height, width=width, generator=generator,
        num_inference_steps=num_inference_steps, output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    generator = torch.manual_seed(0)
    no_res_bin_image = pipe(prompt, height=height, width=width, generator=
        generator, num_inference_steps=num_inference_steps, output_type=
        'np', use_resolution_binning=False).images
    no_res_bin_image_slice = no_res_bin_image[0, -3:, -3:, -1]
    assert not np.allclose(image_slice, no_res_bin_image_slice, atol=0.0001,
        rtol=0.0001)

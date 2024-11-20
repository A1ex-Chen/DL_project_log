def test_single_file_format_inference_is_same_as_pretrained(self):
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png'
        )
    prompt = 'a cat sitting on a park bench'
    pipe = StableDiffusionUpscalePipeline.from_pretrained(self.repo_id)
    pipe.enable_model_cpu_offload()
    generator = torch.Generator('cpu').manual_seed(0)
    output = pipe(prompt=prompt, image=image, generator=generator,
        output_type='np', num_inference_steps=3)
    image_from_pretrained = output.images[0]
    pipe_from_single_file = StableDiffusionUpscalePipeline.from_single_file(
        self.ckpt_path)
    pipe_from_single_file.enable_model_cpu_offload()
    generator = torch.Generator('cpu').manual_seed(0)
    output_from_single_file = pipe_from_single_file(prompt=prompt, image=
        image, generator=generator, output_type='np', num_inference_steps=3)
    image_from_single_file = output_from_single_file.images[0]
    assert image_from_pretrained.shape == (512, 512, 3)
    assert image_from_single_file.shape == (512, 512, 3)
    assert numpy_cosine_similarity_distance(image_from_pretrained.flatten(),
        image_from_single_file.flatten()) < 0.001

def test_single_file_format_inference_is_same_as_pretrained(self):
    init_image = load_image(
        'https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_img2img/sketch-mountains-input.png'
        )
    pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=
        torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.set_default_attn_processor()
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    image = pipe(prompt='mountains', image=init_image, num_inference_steps=
        5, generator=generator, output_type='np').images[0]
    pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path,
        torch_dtype=torch.float16)
    pipe_single_file.scheduler = DDIMScheduler.from_config(pipe_single_file
        .scheduler.config)
    pipe_single_file.unet.set_default_attn_processor()
    pipe_single_file.enable_model_cpu_offload()
    generator = torch.Generator(device='cpu').manual_seed(0)
    image_single_file = pipe_single_file(prompt='mountains', image=
        init_image, num_inference_steps=5, generator=generator, output_type
        ='np').images[0]
    max_diff = numpy_cosine_similarity_distance(image.flatten(),
        image_single_file.flatten())
    assert max_diff < 0.0005

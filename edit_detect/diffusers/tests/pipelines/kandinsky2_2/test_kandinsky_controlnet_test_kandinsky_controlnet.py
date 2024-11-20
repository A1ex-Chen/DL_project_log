def test_kandinsky_controlnet(self):
    expected_image = load_numpy(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/kandinskyv22_controlnet_robotcat_fp16.npy'
        )
    hint = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/hint_image_cat.png'
        )
    hint = torch.from_numpy(np.array(hint)).float() / 255.0
    hint = hint.permute(2, 0, 1).unsqueeze(0)
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior', torch_dtype=torch.float16)
    pipe_prior.enable_model_cpu_offload()
    pipeline = KandinskyV22ControlnetPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-controlnet-depth', torch_dtype=
        torch.float16)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=None)
    prompt = 'A robot, 4k photo'
    generator = torch.Generator(device='cpu').manual_seed(0)
    image_emb, zero_image_emb = pipe_prior(prompt, generator=generator,
        num_inference_steps=2, negative_prompt='').to_tuple()
    generator = torch.Generator(device='cpu').manual_seed(0)
    output = pipeline(image_embeds=image_emb, negative_image_embeds=
        zero_image_emb, hint=hint, generator=generator, num_inference_steps
        =2, output_type='np')
    image = output.images[0]
    assert image.shape == (512, 512, 3)
    max_diff = numpy_cosine_similarity_distance(expected_image.flatten(),
        image.flatten())
    assert max_diff < 0.0001

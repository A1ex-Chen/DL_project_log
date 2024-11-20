def test_stable_cascade_decoder_single_prompt_multiple_image_embeddings_with_guidance(
    self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableCascadeDecoderPipeline(**components)
    pipe.set_progress_bar_config(disable=None)
    prior_num_images_per_prompt = 2
    decoder_num_images_per_prompt = 2
    prompt = ['a cat']
    batch_size = len(prompt)
    generator = torch.Generator(device)
    image_embeddings = randn_tensor((batch_size *
        prior_num_images_per_prompt, 4, 4, 4), generator=generator.
        manual_seed(0))
    decoder_output = pipe(image_embeddings=image_embeddings, prompt=prompt,
        num_inference_steps=1, output_type='np', guidance_scale=2.0,
        generator=generator.manual_seed(0), num_images_per_prompt=
        decoder_num_images_per_prompt)
    assert decoder_output.images.shape[0
        ] == batch_size * prior_num_images_per_prompt * decoder_num_images_per_prompt

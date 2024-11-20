def test_stable_cascade_decoder_prompt_embeds(self):
    device = 'cpu'
    components = self.get_dummy_components()
    pipe = StableCascadeDecoderPipeline(**components)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image_embeddings = inputs['image_embeddings']
    prompt = 'A photograph of a shiba inu, wearing a hat'
    (prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds,
        negative_prompt_embeds_pooled) = pipe.encode_prompt(device, 1, 1, 
        False, prompt=prompt)
    generator = torch.Generator(device=device)
    decoder_output_prompt = pipe(image_embeddings=image_embeddings, prompt=
        prompt, num_inference_steps=1, output_type='np', generator=
        generator.manual_seed(0))
    decoder_output_prompt_embeds = pipe(image_embeddings=image_embeddings,
        prompt=None, prompt_embeds=prompt_embeds, prompt_embeds_pooled=
        prompt_embeds_pooled, negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
        num_inference_steps=1, output_type='np', generator=generator.
        manual_seed(0))
    assert np.abs(decoder_output_prompt.images -
        decoder_output_prompt_embeds.images).max() < 1e-05

def get_dummy_inputs(self, device, seed=0):
    image_embeds = floats_tensor((1, self.text_embedder_hidden_size), rng=
        random.Random(seed)).to(device)
    negative_image_embeds = floats_tensor((1, self.
        text_embedder_hidden_size), rng=random.Random(seed + 1)).to(device)
    hint = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(device)
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'image_embeds': image_embeds, 'negative_image_embeds':
        negative_image_embeds, 'hint': hint, 'generator': generator,
        'height': 64, 'width': 64, 'guidance_scale': 4.0,
        'num_inference_steps': 2, 'output_type': 'np'}
    return inputs

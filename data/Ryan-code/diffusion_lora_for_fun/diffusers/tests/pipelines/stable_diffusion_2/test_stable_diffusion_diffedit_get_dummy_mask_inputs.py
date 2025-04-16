def get_dummy_mask_inputs(self, device, seed=0):
    image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
    image = image.cpu().permute(0, 2, 3, 1)[0]
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    if str(device).startswith('mps'):
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)
    inputs = {'image': image, 'source_prompt': 'a cat and a frog',
        'target_prompt': 'a dog and a newt', 'generator': generator,
        'num_inference_steps': 2, 'num_maps_per_mask': 2,
        'mask_encode_strength': 1.0, 'guidance_scale': 6.0, 'output_type': 'np'
        }
    return inputs

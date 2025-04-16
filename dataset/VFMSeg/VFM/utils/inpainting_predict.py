@torch.no_grad()
def predict(model, input_image, prompt, ddim_steps, num_samples, scale, seed):
    """_summary_

    Args:
        input_image (_type_): dict
            - image: PIL.Image. Input image.
            - mask: PIL.Image. Mask image.
        prompt (_type_): string to be used as prompt. 
        ddim_steps (_type_): typical 45
        num_samples (_type_): typical 4
        scale (_type_): typical 10.0 Guidance Scale.
        seed (_type_): typical 1529160519
    
    """
    init_image = input_image['image'].convert('RGB')
    init_mask = input_image['mask'].convert('RGB')
    image = pad_image(init_image)
    mask = pad_image(init_mask)
    width, height = image.size
    print('Inpainting...', width, height)
    result = inpaint(sampler=model, image=image, mask=mask, prompt=prompt,
        seed=seed, scale=scale, ddim_steps=ddim_steps, num_samples=
        num_samples, h=height, w=width)
    return result

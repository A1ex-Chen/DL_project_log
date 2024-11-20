def save_model_card(repo_id: str, images=None, base_model=str, prompt=str,
    repo_folder=None):
    img_str = ''
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f'image_{i}.png'))
        img_str += f'![img_{i}](./image_{i}.png)\n'
    model_description = f"""
# Custom Diffusion - {repo_id}

These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. 

{img_str}


For more details on the training, please follow [this link](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion).
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='creativeml-openrail-m', base_model=
        base_model, prompt=prompt, model_description=model_description,
        inference=True)
    tags = ['text-to-image', 'diffusers', 'stable-diffusion',
        'stable-diffusion-diffusers', 'custom-diffusion', 'diffusers-training']
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

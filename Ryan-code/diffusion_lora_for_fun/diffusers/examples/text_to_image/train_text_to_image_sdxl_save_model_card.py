def save_model_card(repo_id: str, images: list=None, validation_prompt: str
    =None, base_model: str=None, dataset_name: str=None, repo_folder: str=
    None, vae_path: str=None):
    img_str = ''
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f'image_{i}.png'))
            img_str += f'![img_{i}](./image_{i}.png)\n'
    model_description = f"""
# Text-to-image finetuning - {repo_id}

This pipeline was finetuned from **{base_model}** on the **{dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompt: {validation_prompt}: 

{img_str}

Special VAE used for training: {vae_path}.
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='creativeml-openrail-m', base_model=
        base_model, model_description=model_description, inference=True)
    tags = ['stable-diffusion-xl', 'stable-diffusion-xl-diffusers',
        'text-to-image', 'diffusers-training', 'diffusers']
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

def save_model_card(repo_id: str, images: list=None, base_model: str=None,
    dataset_name: str=None, train_text_encoder: bool=False, repo_folder:
    str=None, vae_path: str=None):
    img_str = ''
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f'image_{i}.png'))
            img_str += f'![img_{i}](./image_{i}.png)\n'
    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. 

{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='creativeml-openrail-m', base_model=
        base_model, model_description=model_description, inference=True)
    tags = ['stable-diffusion-xl', 'stable-diffusion-xl-diffusers',
        'text-to-image', 'diffusers', 'diffusers-training', 'lora']
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

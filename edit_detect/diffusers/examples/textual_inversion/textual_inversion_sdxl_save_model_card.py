def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None
    ):
    img_str = ''
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f'image_{i}.png'))
        img_str += f'![img_{i}](./image_{i}.png)\n'
    model_description = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. 

{img_str}
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='creativeml-openrail-m', base_model=
        base_model, model_description=model_description, inference=True)
    tags = ['stable-diffusion-xl', 'stable-diffusion-xl-diffusers',
        'text-to-image', 'diffusers', 'diffusers-training', 'textual_inversion'
        ]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

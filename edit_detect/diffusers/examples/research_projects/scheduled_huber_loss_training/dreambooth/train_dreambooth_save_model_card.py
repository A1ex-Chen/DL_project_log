def save_model_card(repo_id: str, images: list=None, base_model: str=None,
    train_text_encoder=False, prompt: str=None, repo_folder: str=None,
    pipeline: DiffusionPipeline=None):
    img_str = ''
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f'image_{i}.png'))
            img_str += f'![img_{i}](./image_{i}.png)\n'
    model_description = f"""
# DreamBooth - {repo_id}

This is a dreambooth model derived from {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/).
You can find some example images in the following. 

{img_str}

DreamBooth for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='creativeml-openrail-m', base_model=
        base_model, prompt=prompt, model_description=model_description,
        inference=True)
    tags = ['text-to-image', 'dreambooth', 'diffusers-training']
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(['stable-diffusion', 'stable-diffusion-diffusers'])
    else:
        tags.extend(['if', 'if-diffusers'])
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

def save_model_card(repo_id: str, images=None, base_model=str,
    train_text_encoder=False, prompt=str, repo_folder=None, pipeline:
    DiffusionPipeline=None):
    img_str = ''
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f'image_{i}.png'))
        img_str += f'![img_{i}](./image_{i}.png)\n'
    model_description = f"""
# LoRA DreamBooth - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. 

{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='creativeml-openrail-m', base_model=
        base_model, prompt=prompt, model_description=model_description,
        inference=True)
    tags = ['text-to-image', 'diffusers', 'lora', 'diffusers-training']
    if isinstance(pipeline, StableDiffusionPipeline):
        tags.extend(['stable-diffusion', 'stable-diffusion-diffusers'])
    else:
        tags.extend(['if', 'if-diffusers'])
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

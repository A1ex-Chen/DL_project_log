def save_model_card(repo_id: str, image_logs=None, base_model=str,
    repo_folder=None):
    img_str = ''
    if image_logs is not None:
        img_str = 'You can find some example images below.\n\n'
        for i, log in enumerate(image_logs):
            images = log['images']
            validation_prompt = log['validation_prompt']
            validation_image = log['validation_image']
            validation_image.save(os.path.join(repo_folder,
                'image_control.png'))
            img_str += f'prompt: {validation_prompt}\n'
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(
                repo_folder, f'images_{i}.png'))
            img_str += f'![images_{i})](./images_{i}.png)\n'
    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='openrail++', base_model=base_model,
        model_description=model_description, inference=True)
    tags = ['stable-diffusion-xl', 'stable-diffusion-xl-diffusers',
        'text-to-image', 'diffusers', 'controlnet', 'diffusers-training']
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

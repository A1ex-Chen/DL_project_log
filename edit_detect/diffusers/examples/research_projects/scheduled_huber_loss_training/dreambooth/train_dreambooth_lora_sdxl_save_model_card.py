def save_model_card(repo_id: str, use_dora: bool, images=None, base_model:
    str=None, train_text_encoder=False, instance_prompt=None,
    validation_prompt=None, repo_folder=None, vae_path=None):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f'image_{i}.png'))
            widget_dict.append({'text': validation_prompt if
                validation_prompt else ' ', 'output': {'url':
                f'image_{i}.png'}})
    model_description = f"""
# {'SDXL' if 'playground' not in base_model else 'Playground'} LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

These are {repo_id} LoRA adaption weights for {base_model}.

The weights were trained  using [DreamBooth](https://dreambooth.github.io/).

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## Trigger words

You should use {instance_prompt} to trigger the image generation.

## Download model

Weights for this model are available in Safetensors format.

[Download]({repo_id}/tree/main) them in the Files & versions tab.

"""
    if 'playground' in base_model:
        model_description += """

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(repo_id_or_path=repo_id,
        from_training=True, license='openrail++' if 'playground' not in
        base_model else 'playground-v2dot5-community', base_model=
        base_model, prompt=instance_prompt, model_description=
        model_description, widget=widget_dict)
    tags = ['text-to-image', 'text-to-image', 'diffusers-training',
        'diffusers', 'lora' if not use_dora else 'dora', 'template:sd-lora']
    if 'playground' in base_model:
        tags.extend(['playground', 'playground-diffusers'])
    else:
        tags.extend(['stable-diffusion-xl', 'stable-diffusion-xl-diffusers'])
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, 'README.md'))

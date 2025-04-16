def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None
    ):
    img_str = ''
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f'image_{i}.png'))
        img_str += f'![img_{i}](./image_{i}.png)\n'
    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
- diffusers-training
- onxruntime
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. 

{img_str}
"""
    with open(os.path.join(repo_folder, 'README.md'), 'w') as f:
        f.write(yaml + model_card)

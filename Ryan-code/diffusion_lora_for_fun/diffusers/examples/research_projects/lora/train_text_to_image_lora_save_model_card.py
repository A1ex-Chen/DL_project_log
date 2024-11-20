def save_model_card(repo_id: str, images=None, base_model=str, dataset_name
    =str, repo_folder=None):
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
- diffusers-training
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. 

{img_str}
"""
    with open(os.path.join(repo_folder, 'README.md'), 'w') as f:
        f.write(yaml + model_card)

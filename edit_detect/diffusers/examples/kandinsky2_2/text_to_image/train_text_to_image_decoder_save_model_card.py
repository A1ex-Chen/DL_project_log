def save_model_card(args, repo_id: str, images=None, repo_folder=None):
    img_str = ''
    if len(images) > 0:
        image_grid = make_image_grid(images, 1, len(args.validation_prompts))
        image_grid.save(os.path.join(repo_folder, 'val_imgs_grid.png'))
        img_str += '![val_imgs_grid](./val_imgs_grid.png)\n'
    yaml = f"""
---
license: creativeml-openrail-m
base_model: {args.pretrained_decoder_model_name_or_path}
datasets:
- {args.dataset_name}
prior:
- {args.pretrained_prior_model_name_or_path}
tags:
- kandinsky
- text-to-image
- diffusers
- diffusers-training
inference: true
---
    """
    model_card = f"""
# Finetuning - {repo_id}

This pipeline was finetuned from **{args.pretrained_decoder_model_name_or_path}** on the **{args.dataset_name}** dataset. Below are some example images generated with the finetuned pipeline using the following prompts: {args.validation_prompts}: 

{img_str}

## Pipeline usage

You can use the pipeline like so:

```python
from diffusers import DiffusionPipeline
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("{repo_id}", torch_dtype=torch.float16)
prompt = "{args.validation_prompts[0]}"
image = pipeline(prompt).images[0]
image.save("my_image.png")
```

## Training info

These are the key hyperparameters used during training:

* Epochs: {args.num_train_epochs}
* Learning rate: {args.learning_rate}
* Batch size: {args.train_batch_size}
* Gradient accumulation steps: {args.gradient_accumulation_steps}
* Image resolution: {args.resolution}
* Mixed-precision: {args.mixed_precision}

"""
    wandb_info = ''
    if is_wandb_available():
        wandb_run_url = None
        if wandb.run is not None:
            wandb_run_url = wandb.run.url
    if wandb_run_url is not None:
        wandb_info = f"""
More information on all the CLI arguments and the environment are available on your [`wandb` run page]({wandb_run_url}).
"""
    model_card += wandb_info
    with open(os.path.join(repo_folder, 'README.md'), 'w') as f:
        f.write(yaml + model_card)

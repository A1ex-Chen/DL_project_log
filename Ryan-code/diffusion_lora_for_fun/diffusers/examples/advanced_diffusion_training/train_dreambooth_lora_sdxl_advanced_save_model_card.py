def save_model_card(repo_id: str, use_dora: bool, images=None, base_model=
    str, train_text_encoder=False, train_text_encoder_ti=False,
    token_abstraction_dict=None, instance_prompt=str, validation_prompt=str,
    repo_folder=None, vae_path=None):
    img_str = 'widget:\n'
    lora = 'lora' if not use_dora else 'dora'
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f'image_{i}.png'))
        img_str += f"""
        - text: '{validation_prompt if validation_prompt else ' '}'
          output:
            url:
                "image_{i}.png"
        """
    if not images:
        img_str += f"""
        - text: '{instance_prompt}'
        """
    embeddings_filename = f'{repo_folder}_emb'
    instance_prompt_webui = re.sub('<s\\d+>', '', re.sub('<s\\d+>',
        embeddings_filename, instance_prompt, count=1))
    ti_keys = ', '.join(f'"{match}"' for match in re.findall('<s\\d+>',
        instance_prompt))
    if instance_prompt_webui != embeddings_filename:
        instance_prompt_sentence = f'For example, `{instance_prompt_webui}`'
    else:
        instance_prompt_sentence = ''
    trigger_str = (
        f'You should use {instance_prompt} to trigger the image generation.')
    diffusers_imports_pivotal = ''
    diffusers_example_pivotal = ''
    webui_example_pivotal = ''
    if train_text_encoder_ti:
        trigger_str = """To trigger image generation of trained concept(or concepts) replace each concept identifier in you prompt with the new inserted tokens:
"""
        diffusers_imports_pivotal = """from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
        """
        diffusers_example_pivotal = f"""embedding_path = hf_hub_download(repo_id='{repo_id}', filename='{embeddings_filename}.safetensors', repo_type="model")
state_dict = load_file(embedding_path)
pipeline.load_textual_inversion(state_dict["clip_l"], token=[{ti_keys}], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
pipeline.load_textual_inversion(state_dict["clip_g"], token=[{ti_keys}], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)
        """
        webui_example_pivotal = f"""- *Embeddings*: download **[`{embeddings_filename}.safetensors` here 💾](/{repo_id}/blob/main/{embeddings_filename}.safetensors)**.
    - Place it on it on your `embeddings` folder
    - Use it by adding `{embeddings_filename}` to your prompt. {instance_prompt_sentence}
    (you need both the LoRA and the embeddings as they were trained together for this LoRA)
    """
        if token_abstraction_dict:
            for key, value in token_abstraction_dict.items():
                tokens = ''.join(value)
                trigger_str += (
                    f'\nto trigger concept `{key}` → use `{tokens}` in your prompt \n\n'
                    )
    yaml = f"""---
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- diffusers-training
- text-to-image
- diffusers
- {lora}
- template:sd-lora
{img_str}
base_model: {base_model}
instance_prompt: {instance_prompt}
license: openrail++
---
"""
    model_card = f"""
# SDXL LoRA DreamBooth - {repo_id}

<Gallery />

## Model description

### These are {repo_id} LoRA adaption weights for {base_model}.

## Download model

### Use it with UIs such as AUTOMATIC1111, Comfy UI, SD.Next, Invoke

- **LoRA**: download **[`{repo_folder}.safetensors` here 💾](/{repo_id}/blob/main/{repo_folder}.safetensors)**.
    - Place it on your `models/Lora` folder.
    - On AUTOMATIC1111, load the LoRA by adding `<lora:{repo_folder}:1>` to your prompt. On ComfyUI just [load it as a regular LoRA](https://comfyanonymous.github.io/ComfyUI_examples/lora/).
{webui_example_pivotal}

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
{diffusers_imports_pivotal}
pipeline = AutoPipelineForText2Image.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
{diffusers_example_pivotal}
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## Trigger words

{trigger_str}

## Details
All [Files & versions](/{repo_id}/tree/main).

The weights were trained using [🧨 diffusers Advanced Dreambooth Training Script](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py).

LoRA for the text encoder was enabled. {train_text_encoder}.

Pivotal tuning was enabled: {train_text_encoder_ti}.

Special VAE used for training: {vae_path}.

"""
    with open(os.path.join(repo_folder, 'README.md'), 'w') as f:
        f.write(yaml + model_card)

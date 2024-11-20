def log_validation(pipeline, text_encoder, unet, val_pairs, accelerator):
    pipeline.text_encoder = accelerator.unwrap_model(text_encoder)
    pipeline.unet = accelerator.unwrap_model(unet)
    with torch.autocast('cuda'):
        val_results = [{'data_or_path': pipeline(**pair).images[0],
            'caption': pair['prompt']} for pair in val_pairs]
    torch.cuda.empty_cache()
    wandb.log({'validation': [wandb.Image(**val_result) for val_result in
        val_results]})

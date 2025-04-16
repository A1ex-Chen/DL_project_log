def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        for model in models:
            sub_dir = 'unet' if isinstance(model.base_model.model, type(
                accelerator.unwrap_model(unet).base_model.model)
                ) else 'text_encoder'
            model.save_pretrained(os.path.join(output_dir, sub_dir))
            weights.pop()

def load_model_hook(models, input_dir):
    unet_ = accelerator.unwrap_model(unet)
    unet_.load_adapter(input_dir, 'default', is_trainable=True)
    for _ in range(len(models)):
        models.pop()

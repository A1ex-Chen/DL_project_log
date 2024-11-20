def unfuse_text_encoder_lora(text_encoder):
    for module in text_encoder.modules():
        if isinstance(module, BaseTunerLayer):
            module.unmerge()

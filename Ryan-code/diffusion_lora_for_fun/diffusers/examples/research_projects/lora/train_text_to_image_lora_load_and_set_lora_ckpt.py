def load_and_set_lora_ckpt(pipe, ckpt_dir, global_step, device, dtype):
    with open(os.path.join(args.output_dir,
        f'{global_step}_lora_config.json'), 'r') as f:
        lora_config = json.load(f)
    print(lora_config)
    checkpoint = os.path.join(args.output_dir, f'{global_step}_lora.pt')
    lora_checkpoint_sd = torch.load(checkpoint)
    unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if 
        'text_encoder_' not in k}
    text_encoder_lora_ds = {k.replace('text_encoder_', ''): v for k, v in
        lora_checkpoint_sd.items() if 'text_encoder_' in k}
    unet_config = LoraConfig(**lora_config['peft_config'])
    pipe.unet = LoraModel(unet_config, pipe.unet)
    set_peft_model_state_dict(pipe.unet, unet_lora_ds)
    if 'text_encoder_peft_config' in lora_config:
        text_encoder_config = LoraConfig(**lora_config[
            'text_encoder_peft_config'])
        pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)
    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()
    pipe.to(device)
    return pipe

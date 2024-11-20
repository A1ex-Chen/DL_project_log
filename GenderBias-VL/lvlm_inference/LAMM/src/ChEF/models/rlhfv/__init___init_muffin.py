def init_muffin(model_path, device=None):
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Load muffin model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_config(model_name)
    model = Beit3LlavaLlamaForCausalLM.from_pretrained(model_name,
        torch_dtype=torch.float16, device_map={'': device})
    image_processor = build_transform(is_train=False, input_size=model.
        model.vision_tower.args.img_size)
    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
            special_tokens=True)
    vision_tower = model.model.vision_tower
    if device is not None:
        vision_tower.to(device=device, dtype=torch.float16)
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([
        DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = (tokenizer
            .convert_tokens_to_ids([DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN]))
    image_token_len = model.model.config.num_query
    return model, image_processor, image_token_len, tokenizer

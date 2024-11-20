def load(name, llama_dir, device='cuda' if torch.cuda.is_available() else
    'cpu', download_root='ckpts', max_seq_len=512, max_batch_size=1, phase=
    'finetune'):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(
            f'Model {name} not found; available models = {available_models()}'
            ), None
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')
    model_cfg = ckpt.get('config', {})
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path, max_seq_len
        =max_seq_len, max_batch_size=max_batch_size, clip_model='ViT-L/14',
        v_embed_dim=768, v_depth=8, v_num_heads=16, v_mlp_ratio=4.0,
        query_len=10, query_layer=31, w_bias=model_cfg.get('w_bias', False),
        w_lora=model_cfg.get('w_lora', False), lora_rank=model_cfg.get(
        'lora_rank', 16), w_new_gate=model_cfg.get('w_lora', False), phase=
        phase)
    load_result = model.load_state_dict(ckpt['model'], strict=False)
    assert len(load_result.unexpected_keys
        ) == 0, f'Unexpected keys: {load_result.unexpected_keys}'
    return model.to(device), model.clip_transform

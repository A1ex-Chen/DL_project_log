@classmethod
def from_config(cls, cfg):
    vit_model = cfg.get('vit_model', 'eva_clip_g')
    img_size = cfg.get('image_size')
    llama_model = cfg.get('llama_model')
    drop_path_rate = cfg.get('drop_path_rate', 0)
    use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)
    vit_precision = cfg.get('vit_precision', 'fp16')
    freeze_vit = cfg.get('freeze_vit', True)
    low_resource = cfg.get('low_resource', False)
    prompt_template = cfg.get('prompt_template', '[INST] {} [/INST]')
    max_txt_len = cfg.get('max_txt_len', 300)
    end_sym = cfg.get('end_sym', '\n')
    lora_r = cfg.get('lora_r', 64)
    lora_alpha = cfg.get('lora_alpha', 16)
    chat_template = cfg.get('chat_template', False)
    use_grad_checkpoint_llm = cfg.get('use_grad_checkpoint_llm', False)
    max_context_len = cfg.get('max_context_len', 3800)
    model = cls(vit_model=vit_model, img_size=img_size, drop_path_rate=
        drop_path_rate, use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision, freeze_vit=freeze_vit, llama_model=
        llama_model, prompt_template=prompt_template, max_txt_len=
        max_txt_len, low_resource=low_resource, end_sym=end_sym, lora_r=
        lora_r, lora_alpha=lora_alpha, chat_template=chat_template,
        use_grad_checkpoint_llm=use_grad_checkpoint_llm, max_context_len=
        max_context_len)
    ckpt_path = cfg.get('ckpt', '')
    if ckpt_path:
        print('Load Minigpt-4-LLM Checkpoint: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        msg = model.load_state_dict(ckpt['model'], strict=False)
    return model

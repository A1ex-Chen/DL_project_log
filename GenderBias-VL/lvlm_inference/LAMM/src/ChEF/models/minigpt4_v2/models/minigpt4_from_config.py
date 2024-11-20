@classmethod
def from_config(cls, cfg):
    vit_model = cfg.get('vit_model', 'eva_clip_g')
    q_former_model = cfg.get('q_former_model',
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth'
        )
    img_size = cfg.get('image_size')
    num_query_token = cfg.get('num_query_token')
    llama_model = cfg.get('llama_model')
    drop_path_rate = cfg.get('drop_path_rate', 0)
    use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)
    vit_precision = cfg.get('vit_precision', 'fp16')
    freeze_vit = cfg.get('freeze_vit', True)
    has_qformer = cfg.get('has_qformer', True)
    freeze_qformer = cfg.get('freeze_qformer', True)
    low_resource = cfg.get('low_resource', False)
    device_8bit = cfg.get('device_8bit', 0)
    prompt_path = cfg.get('prompt_path', '')
    prompt_template = cfg.get('prompt_template', '')
    max_txt_len = cfg.get('max_txt_len', 32)
    end_sym = cfg.get('end_sym', '\n')
    model = cls(vit_model=vit_model, q_former_model=q_former_model,
        img_size=img_size, drop_path_rate=drop_path_rate,
        use_grad_checkpoint=use_grad_checkpoint, vit_precision=
        vit_precision, freeze_vit=freeze_vit, has_qformer=has_qformer,
        freeze_qformer=freeze_qformer, num_query_token=num_query_token,
        llama_model=llama_model, prompt_path=prompt_path, prompt_template=
        prompt_template, max_txt_len=max_txt_len, end_sym=end_sym,
        low_resource=low_resource, device_8bit=device_8bit)
    ckpt_path = cfg.get('ckpt', '')
    if ckpt_path:
        print('Load MiniGPT-4 Checkpoint: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        msg = model.load_state_dict(ckpt['model'], strict=False)
    return model

@classmethod
def from_config(cls, cfg):
    vit_model = cfg.get('vit_model', 'eva_clip_g')
    img_size = cfg.get('image_size')
    num_query_token = cfg.get('num_query_token')
    t5_model = cfg.get('t5_model')
    drop_path_rate = cfg.get('drop_path_rate', 0)
    use_grad_checkpoint = cfg.get('use_grad_checkpoint', False)
    vit_precision = cfg.get('vit_precision', 'fp16')
    freeze_vit = cfg.get('freeze_vit', True)
    prompt = cfg.get('prompt', '')
    max_txt_len = cfg.get('max_txt_len', 128)
    max_output_txt_len = cfg.get('max_output_txt_len', 256)
    apply_lemmatizer = cfg.get('apply_lemmatizer', False)
    num_few_shot_examples = cfg.get('num_few_shot_examples', 0)
    few_shot_prob = cfg.get('few_shot_prob', 0.0)
    qformer_text_input = cfg.get('qformer_text_input', True)
    model = cls(vit_model=vit_model, img_size=img_size, drop_path_rate=
        drop_path_rate, use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision, freeze_vit=freeze_vit, num_query_token
        =num_query_token, t5_model=t5_model, prompt=prompt, max_txt_len=
        max_txt_len, max_output_txt_len=max_output_txt_len,
        apply_lemmatizer=apply_lemmatizer, num_few_shot_examples=
        num_few_shot_examples, few_shot_prob=few_shot_prob,
        qformer_text_input=qformer_text_input)
    model.load_checkpoint_from_config(cfg)
    return model

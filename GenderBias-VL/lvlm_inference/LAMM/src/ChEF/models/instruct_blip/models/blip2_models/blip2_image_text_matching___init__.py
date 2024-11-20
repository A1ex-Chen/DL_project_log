def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate=0,
    use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
    num_query_token=32, cross_attention_freq=2, embed_dim=256, max_txt_len=32):
    super().__init__(vit_model=vit_model, img_size=img_size, drop_path_rate
        =drop_path_rate, use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision, freeze_vit=freeze_vit, num_query_token
        =num_query_token, cross_attention_freq=cross_attention_freq,
        embed_dim=embed_dim, max_txt_len=max_txt_len)

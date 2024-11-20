def __init__(self, vit_model='eva_clip_g', img_size=448, drop_path_rate=0,
    use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
    llama_model='', prompt_template='[INST] {} [/INST]', max_txt_len=300,
    end_sym='\n', lora_r=64, lora_target_modules=['q_proj', 'v_proj'],
    lora_alpha=16, lora_dropout=0.05, chat_template=False,
    use_grad_checkpoint_llm=False, max_context_len=3800, low_resource=False,
    device_8bit=0):
    super().__init__(vit_model=vit_model, img_size=img_size, drop_path_rate
        =drop_path_rate, use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision, freeze_vit=freeze_vit, llama_model=
        llama_model, max_txt_len=max_txt_len, max_context_len=
        max_context_len, end_sym=end_sym, prompt_template=prompt_template,
        low_resource=low_resource, device_8bit=device_8bit, lora_r=lora_r,
        lora_target_modules=lora_target_modules, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout)
    img_f_dim = self.visual_encoder.num_features * 4
    self.llama_proj = nn.Linear(img_f_dim, self.llama_model.config.hidden_size)
    self.chat_template = chat_template
    if use_grad_checkpoint_llm:
        self.llama_model.gradient_checkpointing_enable()

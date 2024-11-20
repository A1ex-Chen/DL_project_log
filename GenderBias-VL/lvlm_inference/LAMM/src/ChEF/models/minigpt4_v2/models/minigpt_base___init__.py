def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate=0,
    use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
    llama_model='', max_txt_len=32, max_context_len=3800, prompt_template=
    '', end_sym='\n', low_resource=False, device_8bit=0, lora_r=0,
    lora_target_modules=['q_proj', 'v_proj'], lora_alpha=16, lora_dropout=0.05
    ):
    super().__init__()
    self.llama_model, self.llama_tokenizer = self.init_llm(llama_model_path
        =llama_model, low_resource=low_resource, low_res_device=device_8bit,
        lora_r=lora_r, lora_target_modules=lora_target_modules, lora_alpha=
        lora_alpha, lora_dropout=lora_dropout)
    self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model,
        img_size, drop_path_rate, use_grad_checkpoint, vit_precision,
        freeze_vit)
    self.max_txt_len = max_txt_len
    self.max_context_len = max_context_len
    self.end_sym = end_sym
    self.prompt_template = prompt_template
    self.prompt_list = []

def __init__(self, vit_model='eva_clip_g', q_former_model=
    'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth'
    , img_size=224, drop_path_rate=0, use_grad_checkpoint=False,
    vit_precision='fp16', freeze_vit=True, has_qformer=True, freeze_qformer
    =True, num_query_token=32, llama_model='', prompt_path='',
    prompt_template='', max_txt_len=32, end_sym='\n', low_resource=False,
    device_8bit=0):
    super().__init__(vit_model=vit_model, img_size=img_size, drop_path_rate
        =drop_path_rate, use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision, freeze_vit=freeze_vit, llama_model=
        llama_model, max_txt_len=max_txt_len, end_sym=end_sym, low_resource
        =low_resource, device_8bit=device_8bit)
    self.has_qformer = has_qformer
    if self.has_qformer:
        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token,
            self.visual_encoder.num_features, freeze_qformer)
        self.load_from_pretrained(url_or_filename=q_former_model)
        img_f_dim = self.Qformer.config.hidden_size
        print('Loading Q-Former Done')
    else:
        img_f_dim = self.visual_encoder.num_features * 4
        print('Do not use Q-Former here.')
    self.llama_proj = nn.Linear(img_f_dim, self.llama_model.config.hidden_size)
    if prompt_path:
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        filted_prompts = [raw_prompt for raw_prompt in raw_prompts if 
            '<ImageHere>' in raw_prompt]
        self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
        print('Load {} training prompts'.format(len(self.prompt_list)))
        print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
    else:
        self.prompt_list = []

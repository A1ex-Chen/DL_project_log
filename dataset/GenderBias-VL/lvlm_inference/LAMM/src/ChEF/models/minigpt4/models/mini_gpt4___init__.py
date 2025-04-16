def __init__(self, vit_model='eva_clip_g', q_former_model=
    'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth'
    , img_size=224, drop_path_rate=0, use_grad_checkpoint=False,
    vit_precision='fp32', freeze_vit=True, freeze_qformer=True,
    num_query_token=32, llama_model='', prompt_path='', prompt_template='',
    max_txt_len=32, end_sym='\n', low_resource=False, device_8bit=0):
    super().__init__()
    self.tokenizer = self.init_tokenizer()
    self.low_resource = low_resource
    print('Loading VIT')
    self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model,
        img_size, drop_path_rate, use_grad_checkpoint, vit_precision)
    if freeze_vit:
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.ln_vision.train = disabled_train
        logging.info('freeze vision encoder')
    print('Loading VIT Done')
    print('Loading Q-Former')
    self.Qformer, self.query_tokens = self.init_Qformer(num_query_token,
        self.visual_encoder.num_features)
    self.Qformer.cls = None
    self.Qformer.bert.embeddings.word_embeddings = None
    self.Qformer.bert.embeddings.position_embeddings = None
    for layer in self.Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None
    self.load_from_pretrained(url_or_filename=q_former_model)
    if freeze_qformer:
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = disabled_train
        self.query_tokens.requires_grad = False
        logging.info('freeze Qformer')
    print('Loading Q-Former Done')
    print('Loading LLAMA')
    self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model,
        use_fast=False)
    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
    if self.low_resource:
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model,
            torch_dtype=torch.float16, load_in_8bit=True, device_map={'':
            device_8bit})
    else:
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model,
            torch_dtype=torch.float16)
    for name, param in self.llama_model.named_parameters():
        param.requires_grad = False
    print('Loading LLAMA Done')
    self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, self.
        llama_model.config.hidden_size)
    self.max_txt_len = max_txt_len
    self.end_sym = end_sym
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

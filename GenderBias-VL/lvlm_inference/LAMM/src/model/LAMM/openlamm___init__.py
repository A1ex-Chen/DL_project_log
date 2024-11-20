def __init__(self, **args):
    super(LAMMPEFTModel, self).__init__()
    self.args = args
    self.vision_type = args['vision_type'
        ] if 'vision_type' in args else 'image'
    encoder_pretrain = args['encoder_pretrain'
        ] if 'encoder_pretrain' in args else 'clip'
    assert encoder_pretrain in ['clip', 'epcl'
        ], f'Encoder_pretrain: {encoder_pretrain} Not Implemented'
    encoder_ckpt_path = args['encoder_ckpt_path'
        ] if not encoder_pretrain == 'clip' else '~/.cache/clip/ViT-L-14.pt'
    llm_ckpt_path = args['llm_ckpt_path']
    use_system = args['use_system'] if 'use_system' in args else False
    self.conv_template = conversations.conv_templates[args['conv_template']
        ] if 'conv_template' in args else conversations.default_conversation
    self.stage = args['stage']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f'Initializing [{encoder_pretrain}] visual encoder from {encoder_ckpt_path} [{device}]...'
        )
    self.vision_feature_type = args['vision_feature_type']
    self.num_vision_token = args['num_vision_token']
    self.encoder_pretrain = encoder_pretrain
    if self.encoder_pretrain.lower() == 'clip':
        clip_encoder, self.visual_preprocess = load_clip('ViT-L/14', device
            =device)
        self.visual_encoder = clip_encoder.visual
        if self.vision_feature_type == 'global':
            self.vision_hidden_size = 768
            self.num_vision_token = 1
            assert self.num_vision_token == 1, 'Only 1 global token is available!'
        elif self.vision_feature_type == 'local':
            self.vision_hidden_size = 1024
            self.num_vision_token = min(self.num_vision_token, 256)
    elif self.encoder_pretrain.lower() == 'epcl':
        if LOAD_EPCL_EXT is False:
            raise ImportError(
                'Please refer to README.md to install extension for 3D environment.'
                )
        self.use_color = self.args['use_color'
            ] if 'use_color' in self.args else False
        self.use_height = self.args['use_height'
            ] if 'use_height' in self.args else False
        self.num_points = self.args['num_points'
            ] if 'num_points' in self.args else 40000
        if self.vision_feature_type == 'global':
            raise NotImplementedError('Global feature not implemented for EPCL'
                )
        else:
            self.vision_hidden_size = 1024
            self.num_vision_token = self.num_vision_token
        self.visual_encoder = build_epcl_encoder(pretrain=True, store_path=
            encoder_ckpt_path, device=device)
    else:
        raise NotImplementedError(
            f'Encoder {self.encoder_pretrain} not implemented!')
    for name, param in self.visual_encoder.named_parameters():
        param.requires_grad = False
    self.visual_encoder.eval()
    print('Visual encoder initialized.')
    print(f'Initializing language decoder from {llm_ckpt_path} ...')
    self.initialize_language_model(llm_ckpt_path)
    print('Language decoder initialized.')
    self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_ckpt_path,
        use_fast=False)
    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
    self.llama_tokenizer.padding_side = 'right'
    tokens = self.get_special_tokens()
    self.add_tokens(tokens)
    self.build_projection_layer()
    self.max_tgt_len = args['max_tgt_len']
    self.use_system = use_system
    self.use_flash_attn = args.get('use_flash_attn', False)
    self.use_xformers = args.get('use_xformers', False)
    self.device = torch.cuda.current_device()

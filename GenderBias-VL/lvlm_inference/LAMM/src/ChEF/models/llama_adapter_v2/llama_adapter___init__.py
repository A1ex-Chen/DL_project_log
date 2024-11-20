def __init__(self, llama_ckpt_dir, llama_tokenizer, max_seq_len=512,
    max_batch_size=1, clip_model='ViT-L/14', v_embed_dim=768, v_depth=8,
    v_num_heads=16, v_mlp_ratio=4.0, query_len=10, query_layer=31, w_bias=
    False, w_lora=False, lora_rank=16, w_new_gate=False, phase='finetune'):
    super().__init__()
    with open(os.path.join(llama_ckpt_dir, 'params.json'), 'r') as f:
        params = json.loads(f.read())
    w_bias = phase == 'finetune'
    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len,
        max_batch_size=max_batch_size, **params)
    self.clip, self.clip_transform = clip.load(clip_model)
    clip_dim = self.clip.visual.proj.shape[1]
    self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
    self.clip_proj_norm = nn.LayerNorm(v_embed_dim)
    self.query_len = query_len
    self.query_layer = query_layer
    self.visual_query = nn.Embedding(query_len, v_embed_dim)
    self.visual_blocks = nn.ModuleList([Block(v_embed_dim, v_num_heads,
        v_mlp_ratio, qkv_bias=True) for _ in range(v_depth)])
    self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
    self.visual_proj_norm = nn.LayerNorm(model_args.dim)
    self.adapter_query = nn.Embedding(query_len * query_layer, model_args.dim)
    self.tokenizer = Tokenizer(model_path=llama_tokenizer)
    model_args.w_bias = w_bias
    model_args.w_lora = w_lora
    model_args.lora_rank = lora_rank
    model_args.w_new_gate = w_new_gate
    model_args.vocab_size = self.tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    self.llama = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    ckpts = sorted(Path(llama_ckpt_dir).glob('*.pth'))
    for ckpt in ckpts:
        ckpt = torch.load(ckpt, map_location='cpu')
        self.llama.load_state_dict(ckpt, strict=False)
    del self.clip.transformer
    self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    self.phase = phase
    self.get_trainable_params(self.phase)
    for name, param in self.named_parameters():
        if param.requires_grad:
            print(f'Trainable param: {name}, {param.shape}, {param.dtype}')

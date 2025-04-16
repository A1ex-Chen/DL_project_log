def __init__(self, vit_model='eva_clip_g', img_size=224, drop_path_rate=0,
    use_grad_checkpoint=False, vit_precision='fp16', freeze_vit=True,
    num_query_token=32, llm_model='', prompt='', max_txt_len=128,
    max_output_txt_len=256, apply_lemmatizer=False, qformer_text_input=True):
    super().__init__()
    transformers_version = version.parse(transformers.__version__)
    assert transformers_version >= version.parse('4.28'
        ), 'BLIP-2 Vicuna requires transformers>=4.28'
    from transformers import LlamaTokenizer
    from .modeling_llama import LlamaForCausalLM
    self.tokenizer = self.init_tokenizer(truncation_side='left')
    self.visual_encoder, self.ln_vision = self.init_vision_encoder(vit_model,
        img_size, drop_path_rate, use_grad_checkpoint, vit_precision)
    if freeze_vit:
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        logging.info('freeze vision encoder')
    self.Qformer, self.query_tokens = self.init_Qformer(num_query_token,
        self.visual_encoder.num_features)
    if not qformer_text_input:
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
    else:
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
    self.Qformer.cls = None
    self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast
        =False, truncation_side='left')
    self.llm_model = LlamaForCausalLM.from_pretrained(llm_model,
        torch_dtype=torch.float16)
    self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
    self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
    self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
    self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
    for name, param in self.llm_model.named_parameters():
        param.requires_grad = False
    self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.
        llm_model.config.hidden_size)
    self.max_txt_len = max_txt_len
    self.max_output_txt_len = max_output_txt_len
    self.prompt = prompt
    prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors='pt')
    self.prompt_length = prompt_tokens.attention_mask.sum(1)
    self._lemmatizer = None
    self.qformer_text_input = qformer_text_input

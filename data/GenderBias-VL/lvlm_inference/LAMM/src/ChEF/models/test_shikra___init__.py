def __init__(self, model_path, device='cuda', encoder_ckpt_path=None, **kwargs
    ):
    model_args.model_name_or_path = model_path
    if encoder_ckpt_path is not None:
        model_args.vision_tower = encoder_ckpt_path
    training_args.device = device
    model, self.preprocessor = load_pretrained_shikra(model_args, training_args
        )
    model.to(dtype=torch.float16, device=device)
    model.model.vision_tower.to(dtype=torch.float16, device=device)
    self.model = model
    self.preprocessor['target'] = {'boxes': PlainBoxFormatter()}
    self.tokenizer = self.preprocessor['text']
    self.tokenizer.padding_side = 'left'
    self.gen_kwargs = dict(use_cache=True, pad_token_id=self.tokenizer.
        pad_token_id, bos_token_id=self.tokenizer.bos_token_id,
        eos_token_id=self.tokenizer.eos_token_id)
    self.gen_kwargs['do_sample'] = False
    self.device = device

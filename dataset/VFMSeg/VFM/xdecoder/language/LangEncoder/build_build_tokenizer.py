def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get('PRETRAINED_TOKENIZER',
            'openai/clip-vit-base-patch32')
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    elif config_encoder['TOKENIZER'] == 'clip-fast':
        pretrained_tokenizer = config_encoder.get('PRETRAINED_TOKENIZER',
            'openai/clip-vit-base-patch32')
        tokenizer = CLIPTokenizerFast.from_pretrained(pretrained_tokenizer,
            from_slow=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_encoder['TOKENIZER'])
    return tokenizer

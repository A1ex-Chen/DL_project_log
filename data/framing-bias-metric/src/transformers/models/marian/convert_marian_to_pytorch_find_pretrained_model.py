def find_pretrained_model(src_lang: str, tgt_lang: str) ->List[str]:
    """Find models that can accept src_lang as input and return tgt_lang as output."""
    prefix = 'Helsinki-NLP/opus-mt-'
    api = HfApi()
    model_list = api.model_list()
    model_ids = [x.modelId for x in model_list if x.modelId.startswith(
        'Helsinki-NLP')]
    src_and_targ = [remove_prefix(m, prefix).lower().split('-') for m in
        model_ids if '+' not in m]
    matching = [f'{prefix}{a}-{b}' for a, b in src_and_targ if src_lang in
        a and tgt_lang in b]
    return matching

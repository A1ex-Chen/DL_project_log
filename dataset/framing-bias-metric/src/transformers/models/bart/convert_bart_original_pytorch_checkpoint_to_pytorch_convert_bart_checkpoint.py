@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path,
    hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load('pytorch/fairseq', checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path)
    bart.model.upgrade_state_dict(bart.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace('.', '-')
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(
        SAMPLE_TEXT, return_tensors='pt').unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()
    if checkpoint_path == 'bart.large.mnli':
        state_dict = bart.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict['model.shared.weight'] = state_dict[
            'model.decoder.embed_tokens.weight']
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        model = BartForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        fairseq_output = bart.predict('mnli', tokens, return_logits=True)
        new_model_outputs = model(tokens)[0]
    else:
        state_dict = bart.model.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict['shared.weight'] = state_dict['decoder.embed_tokens.weight']
        fairseq_output = bart.extract_features(tokens)
        if hf_checkpoint_name == 'facebook/bart-large':
            model = BartModel(config).eval()
            model.load_state_dict(state_dict)
            new_model_outputs = model(tokens).model[0]
        else:
            model = BartForConditionalGeneration(config).eval()
            model.model.load_state_dict(state_dict)
            if hasattr(model, 'lm_head'):
                model.lm_head = _make_linear_from_emb(model.model.shared)
            new_model_outputs = model.model(tokens)[0]
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)

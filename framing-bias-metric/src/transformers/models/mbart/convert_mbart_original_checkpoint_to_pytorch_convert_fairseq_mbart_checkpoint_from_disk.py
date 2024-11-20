def convert_fairseq_mbart_checkpoint_from_disk(checkpoint_path,
    hf_config_path='facebook/mbart-large-en-ro'):
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict['encoder.embed_tokens.weight'].shape[0]
    mbart_config = MBartConfig.from_pretrained(hf_config_path, vocab_size=
        vocab_size)
    state_dict['shared.weight'] = state_dict['decoder.embed_tokens.weight']
    model = BartForConditionalGeneration(mbart_config)
    model.model.load_state_dict(state_dict)
    return model

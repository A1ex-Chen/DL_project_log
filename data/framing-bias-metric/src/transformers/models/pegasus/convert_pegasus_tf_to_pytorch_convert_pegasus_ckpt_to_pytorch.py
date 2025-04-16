def convert_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str):
    dataset = Path(ckpt_path).parent.name
    desired_max_model_length = task_specific_params[f'summarization_{dataset}'
        ]['max_position_embeddings']
    tok = PegasusTokenizer.from_pretrained('sshleifer/pegasus',
        model_max_length=desired_max_model_length)
    assert tok.model_max_length == desired_max_model_length
    tok.save_pretrained(save_dir)
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    cfg_updates = task_specific_params[f'summarization_{dataset}']
    if dataset == 'large':
        cfg_updates['task_specific_params'] = task_specific_params
    torch_model = convert_pegasus(tf_weights, cfg_updates)
    torch_model.save_pretrained(save_dir)
    sd = torch_model.state_dict()
    sd.pop('model.decoder.embed_positions.weight')
    sd.pop('model.encoder.embed_positions.weight')
    torch.save(sd, Path(save_dir) / 'pytorch_model.bin')

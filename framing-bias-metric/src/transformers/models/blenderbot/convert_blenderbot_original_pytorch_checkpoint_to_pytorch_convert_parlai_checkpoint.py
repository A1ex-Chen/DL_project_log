@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path,
    config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    model = torch.load(checkpoint_path, map_location='cpu')
    sd = model['model']
    cfg = BartConfig.from_json_file(config_json_path)
    m = BartForConditionalGeneration(cfg)
    valid_keys = m.model.state_dict().keys()
    failures = []
    mapping = {}
    for k, v in sd.items():
        if k in IGNORE_KEYS:
            continue
        new_k = rename_state_dict_key(k)
        if new_k not in valid_keys:
            failures.append([k, new_k])
        else:
            mapping[new_k] = v
    if cfg.normalize_before:
        rename_layernorm_keys(sd)
    m.model.load_state_dict(mapping, strict=True)
    m.half()
    m.save_pretrained(pytorch_dump_folder_path)

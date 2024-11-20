@torch.no_grad()
def dump_hf_model(old_ckpt_path: str, new_folder_path: str) ->None:
    old_ckpt = torch.load(old_ckpt_path, map_location='cpu')
    if old_ckpt.get('model', None) is not None:
        old_ckpt = old_ckpt['model']
    new_ckpt = rename_old_checkpoint(old_ckpt)
    config = OtterConfig.from_json_file('otter/config.json')
    model = OtterModel(config)
    model.load_state_dict(new_ckpt, strict=False)
    print(f'Saving HF model to {new_folder_path}')
    model.save_pretrained(new_folder_path)

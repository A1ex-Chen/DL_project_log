@torch.no_grad()
def dump_hf_model(old_ckpt_path: str, new_folder_path: str) ->None:
    os.makedirs(new_folder_path, exist_ok=True)
    old_ckpt = torch.load(old_ckpt_path, map_location='cpu')
    if old_ckpt.get('model', None) is not None:
        old_ckpt = old_ckpt['model']
    config = OtterConfig.from_json_file('otter/config.json')
    model = OtterModel(config)
    new_ckpt = rename_flamingo_checkpoint(old_ckpt)
    model.load_state_dict(new_ckpt, strict=False)
    text_tokenizer = model.text_tokenizer
    text_tokenizer.add_special_tokens({'additional_special_tokens': [
        '<|endofchunk|>', '<image>', '<answer>']})
    model.lang_encoder.resize_token_embeddings(len(text_tokenizer))
    print(f'Saving HF model to {new_folder_path}')
    model.save_pretrained(new_folder_path)

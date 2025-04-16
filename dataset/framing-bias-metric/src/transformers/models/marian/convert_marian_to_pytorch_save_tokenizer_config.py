def save_tokenizer_config(dest_dir: Path):
    dname = dest_dir.name.split('-')
    dct = dict(target_lang=dname[-1], source_lang='-'.join(dname[:-1]))
    save_json(dct, dest_dir / 'tokenizer_config.json')

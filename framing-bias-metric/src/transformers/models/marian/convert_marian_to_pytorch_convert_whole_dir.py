def convert_whole_dir(path=Path('marian_ckpt/')):
    for subdir in tqdm(list(path.ls())):
        dest_dir = f'marian_converted/{subdir.name}'
        if (dest_dir / 'pytorch_model.bin').exists():
            continue
        convert(source_dir, dest_dir)

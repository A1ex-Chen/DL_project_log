def convert_all_sentencepiece_models(model_list=None, repo_path=None,
    dest_dir=Path('marian_converted')):
    """Requires 300GB"""
    save_dir = Path('marian_ckpt')
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    save_paths = []
    if model_list is None:
        model_list: list = make_registry(repo_path=repo_path)
    for k, prepro, download, test_set_url in tqdm(model_list):
        if 'SentencePiece' not in prepro:
            continue
        if not os.path.exists(save_dir / k):
            download_and_unzip(download, save_dir / k)
        pair_name = convert_opus_name_to_hf_name(k)
        convert(save_dir / k, dest_dir / f'opus-mt-{pair_name}')
        save_paths.append(dest_dir / f'opus-mt-{pair_name}')
    return save_paths

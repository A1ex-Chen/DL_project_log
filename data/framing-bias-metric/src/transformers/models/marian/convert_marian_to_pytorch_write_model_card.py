def write_model_card(hf_model_name: str, repo_root=DEFAULT_REPO, save_dir=
    Path('marian_converted'), dry_run=False, extra_metadata={}) ->str:
    """
    Copy the most recent model's readme section from opus, and add metadata. upload command: aws s3 sync model_card_dir
    s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
    """
    import pandas as pd
    hf_model_name = remove_prefix(hf_model_name, ORG_NAME)
    opus_name: str = convert_hf_name_to_opus_name(hf_model_name)
    assert repo_root in ('OPUS-MT-train', 'Tatoeba-Challenge')
    opus_readme_path = Path(repo_root).joinpath('models', opus_name,
        'README.md')
    assert opus_readme_path.exists(
        ), f'Readme file {opus_readme_path} not found'
    opus_src, opus_tgt = [x.split('+') for x in opus_name.split('-')]
    readme_url = (
        f'https://github.com/Helsinki-NLP/{repo_root}/tree/master/models/{opus_name}/README.md'
        )
    s, t = ','.join(opus_src), ','.join(opus_tgt)
    metadata = {'hf_name': hf_model_name, 'source_languages': s,
        'target_languages': t, 'opus_readme_url': readme_url,
        'original_repo': repo_root, 'tags': ['translation']}
    metadata.update(extra_metadata)
    metadata.update(get_system_metadata(repo_root))
    extra_markdown = f"""### {hf_model_name}

* source group: {metadata['src_name']} 
* target group: {metadata['tgt_name']} 
*  OPUS readme: [{opus_name}]({readme_url})
"""
    content = opus_readme_path.open().read()
    content = content.split('\n# ')[-1]
    splat = content.split('*')[2:]
    print(splat[3])
    content = '*'.join(splat)
    content = FRONT_MATTER_TEMPLATE.format(metadata['src_alpha2']
        ) + extra_markdown + '\n* ' + content.replace('download',
        'download original weights')
    items = '\n\n'.join([f'- {k}: {v}' for k, v in metadata.items()])
    sec3 = '\n### System Info: \n' + items
    content += sec3
    if dry_run:
        return content, metadata
    sub_dir = save_dir / f'opus-mt-{hf_model_name}'
    sub_dir.mkdir(exist_ok=True)
    dest = sub_dir / 'README.md'
    dest.open('w').write(content)
    pd.Series(metadata).to_json(sub_dir / 'metadata.json')
    return content, metadata

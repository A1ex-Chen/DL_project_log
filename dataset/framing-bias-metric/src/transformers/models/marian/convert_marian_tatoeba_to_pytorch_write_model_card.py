def write_model_card(self, hf_model_id: str, repo_root=DEFAULT_REPO,
    dry_run=False) ->str:
    """
        Copy the most recent model's readme section from opus, and add metadata. upload command: aws s3 sync
        model_card_dir s3://models.huggingface.co/bert/Helsinki-NLP/ --dryrun
        """
    short_pair = remove_prefix(hf_model_id, 'opus-mt-')
    extra_metadata = self.metadata.loc[short_pair].drop('2m')
    extra_metadata['short_pair'] = short_pair
    lang_tags, src_multilingual, tgt_multilingual = self.resolve_lang_code(
        extra_metadata)
    opus_name = f'{extra_metadata.src_alpha3}-{extra_metadata.tgt_alpha3}'
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
    metadata = {'hf_name': short_pair, 'source_languages': s,
        'target_languages': t, 'opus_readme_url': readme_url,
        'original_repo': repo_root, 'tags': ['translation'], 'languages':
        lang_tags}
    lang_tags = l2front_matter(lang_tags)
    metadata['src_constituents'] = self.constituents[s]
    metadata['tgt_constituents'] = self.constituents[t]
    metadata['src_multilingual'] = src_multilingual
    metadata['tgt_multilingual'] = tgt_multilingual
    metadata.update(extra_metadata)
    metadata.update(get_system_metadata(repo_root))
    extra_markdown = f"""### {short_pair}

* source group: {metadata['src_name']} 
* target group: {metadata['tgt_name']} 
*  OPUS readme: [{opus_name}]({readme_url})
"""
    content = opus_readme_path.open().read()
    content = content.split('\n# ')[-1]
    splat = content.split('*')[2:]
    content = '*'.join(splat)
    content = FRONT_MATTER_TEMPLATE.format(lang_tags
        ) + extra_markdown + '\n* ' + content.replace('download',
        'download original weights')
    items = '\n\n'.join([f'- {k}: {v}' for k, v in metadata.items()])
    sec3 = '\n### System Info: \n' + items
    content += sec3
    if dry_run:
        return content, metadata
    sub_dir = self.model_card_dir / hf_model_id
    sub_dir.mkdir(exist_ok=True)
    dest = sub_dir / 'README.md'
    dest.open('w').write(content)
    pd.Series(metadata).to_json(sub_dir / 'metadata.json')
    return content, metadata

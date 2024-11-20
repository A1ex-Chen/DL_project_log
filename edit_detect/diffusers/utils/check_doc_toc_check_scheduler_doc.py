def check_scheduler_doc(overwrite=False):
    with open(PATH_TO_TOC, encoding='utf-8') as f:
        content = yaml.safe_load(f.read())
    api_idx = 0
    while content[api_idx]['title'] != 'API':
        api_idx += 1
    api_doc = content[api_idx]['sections']
    scheduler_idx = 0
    while api_doc[scheduler_idx]['title'] != 'Schedulers':
        scheduler_idx += 1
    scheduler_doc = api_doc[scheduler_idx]['sections']
    new_scheduler_doc = clean_doc_toc(scheduler_doc)
    diff = False
    if new_scheduler_doc != scheduler_doc:
        diff = True
        if overwrite:
            api_doc[scheduler_idx]['sections'] = new_scheduler_doc
    if diff:
        if overwrite:
            content[api_idx]['sections'] = api_doc
            with open(PATH_TO_TOC, 'w', encoding='utf-8') as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError(
                'The model doc part of the table of content is not properly sorted, run `make style` to fix this.'
                )

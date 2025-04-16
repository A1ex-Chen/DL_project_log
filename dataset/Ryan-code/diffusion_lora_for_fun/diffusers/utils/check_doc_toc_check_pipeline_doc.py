def check_pipeline_doc(overwrite=False):
    with open(PATH_TO_TOC, encoding='utf-8') as f:
        content = yaml.safe_load(f.read())
    api_idx = 0
    while content[api_idx]['title'] != 'API':
        api_idx += 1
    api_doc = content[api_idx]['sections']
    pipeline_idx = 0
    while api_doc[pipeline_idx]['title'] != 'Pipelines':
        pipeline_idx += 1
    diff = False
    pipeline_docs = api_doc[pipeline_idx]['sections']
    new_pipeline_docs = []
    for pipeline_doc in pipeline_docs:
        if 'section' in pipeline_doc:
            sub_pipeline_doc = pipeline_doc['section']
            new_sub_pipeline_doc = clean_doc_toc(sub_pipeline_doc)
            if overwrite:
                pipeline_doc['section'] = new_sub_pipeline_doc
        new_pipeline_docs.append(pipeline_doc)
    new_pipeline_docs = clean_doc_toc(new_pipeline_docs)
    if new_pipeline_docs != pipeline_docs:
        diff = True
        if overwrite:
            api_doc[pipeline_idx]['sections'] = new_pipeline_docs
    if diff:
        if overwrite:
            content[api_idx]['sections'] = api_doc
            with open(PATH_TO_TOC, 'w', encoding='utf-8') as f:
                f.write(yaml.dump(content, allow_unicode=True))
        else:
            raise ValueError(
                'The model doc part of the table of content is not properly sorted, run `make style` to fix this.'
                )

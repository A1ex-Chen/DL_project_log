def clean_doc_toc(doc_list):
    """
    Cleans the table of content of the model documentation by removing duplicates and sorting models alphabetically.
    """
    counts = defaultdict(int)
    overview_doc = []
    new_doc_list = []
    for doc in doc_list:
        if 'local' in doc:
            counts[doc['local']] += 1
        if doc['title'].lower() == 'overview':
            overview_doc.append({'local': doc['local'], 'title': doc['title']})
        else:
            new_doc_list.append(doc)
    doc_list = new_doc_list
    duplicates = [key for key, value in counts.items() if value > 1]
    new_doc = []
    for duplicate_key in duplicates:
        titles = list({doc['title'] for doc in doc_list if doc['local'] ==
            duplicate_key})
        if len(titles) > 1:
            raise ValueError(
                f'{duplicate_key} is present several times in the documentation table of content at `docs/source/en/_toctree.yml` with different *Title* values. Choose one of those and remove the others.'
                )
        new_doc.append({'local': duplicate_key, 'title': titles[0]})
    new_doc.extend([doc for doc in doc_list if 'local' not in counts or 
        counts[doc['local']] == 1])
    new_doc = sorted(new_doc, key=lambda s: s['title'].lower())
    if len(overview_doc) > 1:
        raise ValueError(
            "{doc_list} has two 'overview' docs which is not allowed.")
    overview_doc.extend(new_doc)
    return overview_doc

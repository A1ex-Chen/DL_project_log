def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
    if doc_title.startswith('"'):
        doc_title = doc_title[1:]
    if doc_title.endswith('"'):
        doc_title = doc_title[:-1]
    if prefix is None:
        prefix = ''
    out = (prefix + doc_title + self.config.title_sep + doc_text + self.
        config.doc_sep + input_string).replace('  ', ' ')
    return out

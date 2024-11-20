def add_examples_from_csv(self, file_name, split_name='', column_label=0,
    column_text=1, column_id=None, skip_first_row=False, overwrite_labels=
    False, overwrite_examples=False):
    lines = self._read_tsv(file_name)
    if skip_first_row:
        lines = lines[1:]
    texts = []
    labels = []
    ids = []
    for i, line in enumerate(lines):
        texts.append(line[column_text])
        labels.append(line[column_label])
        if column_id is not None:
            ids.append(line[column_id])
        else:
            guid = '%s-%s' % (split_name, i) if split_name else '%s' % i
            ids.append(guid)
    return self.add_examples(texts, labels, ids, overwrite_labels=
        overwrite_labels, overwrite_examples=overwrite_examples)

def add_examples(self, texts_or_text_and_labels, labels=None, ids=None,
    overwrite_labels=False, overwrite_examples=False):
    assert labels is None or len(texts_or_text_and_labels) == len(labels
        ), f'Text and labels have mismatched lengths {len(texts_or_text_and_labels)} and {len(labels)}'
    assert ids is None or len(texts_or_text_and_labels) == len(ids
        ), f'Text and ids have mismatched lengths {len(texts_or_text_and_labels)} and {len(ids)}'
    if ids is None:
        ids = [None] * len(texts_or_text_and_labels)
    if labels is None:
        labels = [None] * len(texts_or_text_and_labels)
    examples = []
    added_labels = set()
    for text_or_text_and_label, label, guid in zip(texts_or_text_and_labels,
        labels, ids):
        if isinstance(text_or_text_and_label, (tuple, list)) and label is None:
            text, label = text_or_text_and_label
        else:
            text = text_or_text_and_label
        added_labels.add(label)
        examples.append(InputExample(guid=guid, text_a=text, text_b=None,
            label=label))
    if overwrite_examples:
        self.examples = examples
    else:
        self.examples.extend(examples)
    if overwrite_labels:
        self.labels = list(added_labels)
    else:
        self.labels = list(set(self.labels).union(added_labels))
    return self.examples

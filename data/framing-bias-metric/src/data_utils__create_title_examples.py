def _create_title_examples(self, objs, set_type):
    examples = []
    for i, obj in enumerate(objs):
        if i == 0:
            continue
        guid = i
        text_a = obj['title']
        label = obj['label']
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None,
            label=label, task='fnn_buzzfeed_title'))
    return examples

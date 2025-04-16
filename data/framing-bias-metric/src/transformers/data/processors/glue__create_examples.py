def _create_examples(self, lines, set_type):
    """Creates examples for the training, dev and test sets."""
    examples = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        guid = '%s-%s' % (set_type, line[0])
        text_a = line[1]
        text_b = line[2]
        label = None if set_type == 'test' else line[-1]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=
            text_b, label=label))
    return examples

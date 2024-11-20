def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for i, line in enumerate(lines):
        guid = '%s-%s' % (set_type, i)
        for x in line['ents']:
            if x[1] == 1:
                x[1] = 0
        text_a = line['text'], line['ents']
        label = line['label']
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=line[
            'ann'], label=label))
    return examples

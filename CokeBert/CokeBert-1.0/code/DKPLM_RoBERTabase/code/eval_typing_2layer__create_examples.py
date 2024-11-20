def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for i, line in enumerate(lines):
        guid = i
        text_a = line['sent'], [['SPAN', line['start'], line['end']]]
        text_b = line['ents']
        label = line['labels']
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=
            text_b, label=label))
    return examples

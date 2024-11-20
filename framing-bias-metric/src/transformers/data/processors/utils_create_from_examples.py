@classmethod
def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
    processor = cls(**kwargs)
    processor.add_examples(texts_or_text_and_labels, labels=labels)
    return processor

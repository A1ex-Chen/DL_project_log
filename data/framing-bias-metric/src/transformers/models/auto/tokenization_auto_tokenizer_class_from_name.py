def tokenizer_class_from_name(class_name: str):
    all_tokenizer_classes = [v[0] for v in TOKENIZER_MAPPING.values() if v[
        0] is not None] + [v[1] for v in TOKENIZER_MAPPING.values() if v[1]
         is not None] + NO_CONFIG_TOKENIZER
    for c in all_tokenizer_classes:
        if c.__name__ == class_name:
            return c

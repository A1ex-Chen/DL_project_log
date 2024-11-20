def gen():
    for ex in features:
        yield {'input_ids': ex.input_ids, 'attention_mask': ex.attention_mask
            }, ex.label

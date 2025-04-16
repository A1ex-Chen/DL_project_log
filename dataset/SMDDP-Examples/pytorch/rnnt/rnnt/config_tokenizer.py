def tokenizer(conf):
    return validate_and_fill(Tokenizer, conf['tokenizer'], optional=[
        'sentpiece_model'])

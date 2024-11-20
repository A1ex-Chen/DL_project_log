def pad_vocabulary(math):
    if math == 'fp16':
        pad_vocab = 8
    elif math == 'fp32':
        pad_vocab = 1
    return pad_vocab

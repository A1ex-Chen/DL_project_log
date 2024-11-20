def find_vocab_file(model_dir):
    return list(model_dir.glob('*vocab.yml'))[0]

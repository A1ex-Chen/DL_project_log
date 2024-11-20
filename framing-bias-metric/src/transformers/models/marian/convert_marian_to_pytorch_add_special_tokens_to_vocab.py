def add_special_tokens_to_vocab(model_dir: Path) ->None:
    vocab = load_yaml(find_vocab_file(model_dir))
    vocab = {k: int(v) for k, v in vocab.items()}
    num_added = add_to_vocab_(vocab, ['<pad>'])
    print(f'added {num_added} tokens to vocab')
    save_json(vocab, model_dir / 'vocab.json')
    save_tokenizer_config(model_dir)

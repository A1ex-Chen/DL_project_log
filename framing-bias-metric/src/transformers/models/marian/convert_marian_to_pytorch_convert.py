def convert(source_dir: Path, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(exist_ok=True)
    add_special_tokens_to_vocab(source_dir)
    tokenizer = MarianTokenizer.from_pretrained(str(source_dir))
    tokenizer.save_pretrained(dest_dir)
    opus_state = OpusState(source_dir)
    assert opus_state.cfg['vocab_size'] == len(tokenizer.encoder
        ), f"Original vocab size {opus_state.cfg['vocab_size']} and new vocab size {len(tokenizer.encoder)} mismatched"
    model = opus_state.load_marian_model()
    model = model.half()
    model.save_pretrained(dest_dir)
    model.from_pretrained(dest_dir)

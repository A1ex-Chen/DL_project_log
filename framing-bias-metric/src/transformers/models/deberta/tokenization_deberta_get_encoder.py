def get_encoder(encoder, vocab):
    return Encoder(encoder=encoder, bpe_merges=vocab)

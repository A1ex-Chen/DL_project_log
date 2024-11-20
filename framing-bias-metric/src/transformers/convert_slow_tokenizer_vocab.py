def vocab(self, proto):
    num_extra_ids = self.original_tokenizer._extra_ids
    vocab = [(piece.piece, piece.score) for piece in proto.pieces]
    vocab += [('<extra_id_{}>'.format(i), 0.0) for i in range(num_extra_ids -
        1, -1, -1)]
    return vocab

def extract(self) ->Tuple[Dict[str, int], List[Tuple]]:
    sp = self.sp
    vocab = {sp.id_to_piece(index): index for index in range(sp.GetPieceSize())
        }
    merges = []
    for piece_l in vocab.keys():
        for piece_r in vocab.keys():
            merge = f'{piece_l}{piece_r}'
            piece_id = vocab.get(merge, None)
            if piece_id:
                merges += [(piece_l, piece_r, piece_id)]
    merges = sorted(merges, key=lambda val: val[2])
    merges = [(val[0], val[1]) for val in merges]
    return vocab, merges

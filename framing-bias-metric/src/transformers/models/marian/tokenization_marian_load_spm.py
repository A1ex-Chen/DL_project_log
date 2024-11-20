def load_spm(path: str) ->sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(path)
    return spm

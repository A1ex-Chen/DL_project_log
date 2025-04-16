def __setstate__(self, d):
    self.__dict__ = d
    try:
        import sentencepiece as spm
    except ImportError:
        logger.warning(
            'You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiecepip install sentencepiece'
            )
        raise
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(self.vocab_file)

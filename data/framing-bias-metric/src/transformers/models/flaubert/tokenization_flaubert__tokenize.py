def _tokenize(self, text, bypass_tokenizer=False):
    """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:

            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
    lang = 'fr'
    if lang and self.lang2id and lang not in self.lang2id:
        logger.error(
            'Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model.'
            )
    if bypass_tokenizer:
        text = text.split()
    else:
        text = self.preprocess_text(text)
        text = self.moses_pipeline(text, lang=lang)
        text = self.moses_tokenize(text, lang=lang)
    split_tokens = []
    for token in text:
        if token:
            split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens

def _tokenize(self, text, lang='en', bypass_tokenizer=False):
    """
        Tokenize a string given language code using Moses.

        Details of tokenization:

            - [sacremoses](https://github.com/alvations/sacremoses): port of Moses
            - Install with `pip install sacremoses`

        Args:

            - lang: ISO language code (default = 'en') (string). Languages should belong of the model supported
              languages. However, we don't enforce it.
            - bypass_tokenizer: Allow users to preprocess and tokenize the sentences externally (default = False)
              (bool). If True, we only apply BPE.

        Returns:
            List of tokens.
        """
    lang = self.src_lang
    if self.do_lower_case:
        text = text.lower()
    if bypass_tokenizer:
        text = text.split()
    else:
        text = self.moses_pipeline(text, lang=lang)
        text = self.moses_tokenize(text, lang=lang)
    split_tokens = []
    for token in text:
        if token:
            split_tokens.extend([t for t in self.bpe(token).split(' ')])
    return split_tokens

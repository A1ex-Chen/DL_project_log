def casual_tokenize(text, preserve_case=True, reduce_len=False,
    strip_handles=False):
    """
    Convenience function for wrapping the tokenizer.
    """
    return TweetTokenizer(preserve_case=preserve_case, reduce_len=
        reduce_len, strip_handles=strip_handles).tokenize(text)

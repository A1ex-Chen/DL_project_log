@classmethod
def from_pretrained(cls, *args, **kwargs):
    """See `AutoTokenizer.from_pretrained` docstring."""
    tokenizer = super().from_pretrained(*args, **kwargs)
    adapt_tokenizer_for_denoising(tokenizer)
    return tokenizer

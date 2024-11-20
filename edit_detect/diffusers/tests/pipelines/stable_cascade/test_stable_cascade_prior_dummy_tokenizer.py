@property
def dummy_tokenizer(self):
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    return tokenizer

@property
def dummy_tokenizer(self):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        'YiYiXu/tiny-random-mclip-base')
    return tokenizer

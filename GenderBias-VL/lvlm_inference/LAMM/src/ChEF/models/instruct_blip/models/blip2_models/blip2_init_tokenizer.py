@classmethod
def init_tokenizer(cls, truncation_side='right'):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
        truncation_side=truncation_side)
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    return tokenizer

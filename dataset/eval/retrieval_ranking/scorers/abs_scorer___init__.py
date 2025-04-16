def __init__(self, tokenizer):
    self.nlp = spacy.load('en_core_web_sm')
    self.tokenizer = tokenizer

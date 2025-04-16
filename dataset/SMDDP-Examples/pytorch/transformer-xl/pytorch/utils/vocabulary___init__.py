def __init__(self, max_size=None, vocab_file=None):
    from pytorch_transformers import GPT2Tokenizer
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.EOT = self.tokenizer.encoder['<|endoftext|>']
    self.max_size = max_size
    self.vocab_file = vocab_file
    pad = 8
    vocab_size = len(self.tokenizer)
    padded_vocab_size = (vocab_size + pad - 1) // pad * pad
    for i in range(0, padded_vocab_size - vocab_size):
        token = f'madeupword{i:09d}'
        self.tokenizer.add_tokens([token])

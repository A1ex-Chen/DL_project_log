def test_encode_input_type(self):
    tokenizer = self.get_tokenizer()
    sequence = "Let's encode this sequence"
    tokens = tokenizer.tokenize(sequence)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    formatted_input = tokenizer.encode(sequence, add_special_tokens=True)
    self.assertEqual(tokenizer.encode(tokens, add_special_tokens=True),
        formatted_input)
    self.assertEqual(tokenizer.encode(input_ids, add_special_tokens=True),
        formatted_input)

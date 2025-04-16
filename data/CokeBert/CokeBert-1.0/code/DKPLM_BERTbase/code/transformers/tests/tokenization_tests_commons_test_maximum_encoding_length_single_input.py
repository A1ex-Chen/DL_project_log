def test_maximum_encoding_length_single_input(self):
    tokenizer = self.get_tokenizer()
    seq_0 = 'This is a sentence to be encoded.'
    stride = 2
    sequence = tokenizer.encode(seq_0, add_special_tokens=False)
    num_added_tokens = tokenizer.num_added_tokens()
    total_length = len(sequence) + num_added_tokens
    information = tokenizer.encode_plus(seq_0, max_length=total_length - 2,
        add_special_tokens=True, stride=stride)
    truncated_sequence = information['input_ids']
    overflowing_tokens = information['overflowing_tokens']
    self.assertEqual(len(overflowing_tokens), 2 + stride)
    self.assertEqual(overflowing_tokens, sequence[-(2 + stride):])
    self.assertEqual(len(truncated_sequence), total_length - 2)
    self.assertEqual(truncated_sequence, tokenizer.
        build_inputs_with_special_tokens(sequence[:-2]))

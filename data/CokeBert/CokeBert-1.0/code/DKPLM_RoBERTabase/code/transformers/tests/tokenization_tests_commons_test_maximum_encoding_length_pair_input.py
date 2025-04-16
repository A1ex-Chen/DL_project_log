def test_maximum_encoding_length_pair_input(self):
    tokenizer = self.get_tokenizer()
    seq_0 = 'This is a sentence to be encoded.'
    seq_1 = 'This is another sentence to be encoded.'
    stride = 2
    sequence_0_no_special_tokens = tokenizer.encode(seq_0,
        add_special_tokens=False)
    sequence_1_no_special_tokens = tokenizer.encode(seq_1,
        add_special_tokens=False)
    sequence = tokenizer.encode(seq_0, seq_1, add_special_tokens=True)
    truncated_second_sequence = tokenizer.build_inputs_with_special_tokens(
        tokenizer.encode(seq_0, add_special_tokens=False), tokenizer.encode
        (seq_1, add_special_tokens=False)[:-2])
    information = tokenizer.encode_plus(seq_0, seq_1, max_length=len(
        sequence) - 2, add_special_tokens=True, stride=stride,
        truncation_strategy='only_second')
    information_first_truncated = tokenizer.encode_plus(seq_0, seq_1,
        max_length=len(sequence) - 2, add_special_tokens=True, stride=
        stride, truncation_strategy='only_first')
    truncated_sequence = information['input_ids']
    overflowing_tokens = information['overflowing_tokens']
    overflowing_tokens_first_truncated = information_first_truncated[
        'overflowing_tokens']
    self.assertEqual(len(overflowing_tokens), 2 + stride)
    self.assertEqual(overflowing_tokens, sequence_1_no_special_tokens[-(2 +
        stride):])
    self.assertEqual(overflowing_tokens_first_truncated,
        sequence_0_no_special_tokens[-(2 + stride):])
    self.assertEqual(len(truncated_sequence), len(sequence) - 2)
    self.assertEqual(truncated_sequence, truncated_second_sequence)

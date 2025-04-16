def test_number_of_added_tokens(self):
    tokenizer = self.get_tokenizer()
    seq_0 = 'Test this method.'
    seq_1 = 'With these inputs.'
    sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=False)
    attached_sequences = tokenizer.encode(seq_0, seq_1, add_special_tokens=True
        )
    if len(attached_sequences) != 2:
        self.assertEqual(tokenizer.num_added_tokens(pair=True), len(
            attached_sequences) - len(sequences))

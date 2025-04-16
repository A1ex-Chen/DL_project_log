def test_mask_output(self):
    if sys.version_info <= (3, 0):
        return
    tokenizer = self.get_tokenizer()
    if tokenizer.build_inputs_with_special_tokens.__qualname__.split('.')[0
        ] != 'PreTrainedTokenizer':
        seq_0 = 'Test this method.'
        seq_1 = 'With these inputs.'
        information = tokenizer.encode_plus(seq_0, seq_1,
            add_special_tokens=True)
        sequences, mask = information['input_ids'], information[
            'token_type_ids']
        self.assertEqual(len(sequences), len(mask))

def test_special_tokens_mask(self):
    tokenizer = self.get_tokenizer()
    sequence_0 = 'Encode this.'
    sequence_1 = 'This one too please.'
    encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False)
    encoded_sequence_dict = tokenizer.encode_plus(sequence_0,
        add_special_tokens=True)
    encoded_sequence_w_special = encoded_sequence_dict['input_ids']
    special_tokens_mask = encoded_sequence_dict['special_tokens_mask']
    self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
    filtered_sequence = [(x if not special_tokens_mask[i] else None) for i,
        x in enumerate(encoded_sequence_w_special)]
    filtered_sequence = [x for x in filtered_sequence if x is not None]
    self.assertEqual(encoded_sequence, filtered_sequence)
    encoded_sequence = tokenizer.encode(sequence_0, add_special_tokens=False
        ) + tokenizer.encode(sequence_1, add_special_tokens=False)
    encoded_sequence_dict = tokenizer.encode_plus(sequence_0, sequence_1,
        add_special_tokens=True)
    encoded_sequence_w_special = encoded_sequence_dict['input_ids']
    special_tokens_mask = encoded_sequence_dict['special_tokens_mask']
    self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
    filtered_sequence = [(x if not special_tokens_mask[i] else None) for i,
        x in enumerate(encoded_sequence_w_special)]
    filtered_sequence = [x for x in filtered_sequence if x is not None]
    self.assertEqual(encoded_sequence, filtered_sequence)
    if (tokenizer.cls_token_id == tokenizer.unk_token_id and tokenizer.
        cls_token_id == tokenizer.unk_token_id):
        tokenizer.add_special_tokens({'cls_token': '</s>', 'sep_token': '<s>'})
    encoded_sequence_dict = tokenizer.encode_plus(sequence_0,
        add_special_tokens=True)
    encoded_sequence_w_special = encoded_sequence_dict['input_ids']
    special_tokens_mask_orig = encoded_sequence_dict['special_tokens_mask']
    special_tokens_mask = tokenizer.get_special_tokens_mask(
        encoded_sequence_w_special, already_has_special_tokens=True)
    self.assertEqual(len(special_tokens_mask), len(encoded_sequence_w_special))
    self.assertEqual(special_tokens_mask_orig, special_tokens_mask)

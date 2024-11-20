@pytest.mark.slow
def test_sequence_builders(self):
    tokenizer = self.tokenizer_class.from_pretrained('bert-base-uncased')
    text = tokenizer.encode('sequence builders', add_special_tokens=False)
    text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
    encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
    encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
    assert encoded_sentence == [101] + text + [102]
    assert encoded_pair == [101] + text + [102] + text_2 + [102]

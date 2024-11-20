@pytest.mark.slow
def test_sequence_builders(self):
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
    text = tokenizer.encode('sequence builders', add_special_tokens=False)
    text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
    encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
    encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
    assert encoded_sentence == [1] + text + [1]
    assert encoded_pair == [1] + text + [1] + text_2 + [1]

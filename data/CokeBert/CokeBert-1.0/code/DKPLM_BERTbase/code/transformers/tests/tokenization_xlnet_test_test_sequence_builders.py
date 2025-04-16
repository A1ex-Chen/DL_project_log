@pytest.mark.slow
def test_sequence_builders(self):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    text = tokenizer.encode('sequence builders', add_special_tokens=False)
    text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
    encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
    encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
    assert encoded_sentence == text + [4, 3]
    assert encoded_pair == text + [4] + text_2 + [4, 3]

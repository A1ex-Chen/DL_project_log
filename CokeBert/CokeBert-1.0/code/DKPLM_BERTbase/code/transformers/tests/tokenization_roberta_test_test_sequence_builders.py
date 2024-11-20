@pytest.mark.slow
def test_sequence_builders(self):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    text = tokenizer.encode('sequence builders', add_special_tokens=False)
    text_2 = tokenizer.encode('multi-sequence build', add_special_tokens=False)
    encoded_text_from_decode = tokenizer.encode('sequence builders',
        add_special_tokens=True)
    encoded_pair_from_decode = tokenizer.encode('sequence builders',
        'multi-sequence build', add_special_tokens=True)
    encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
    encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)
    assert encoded_sentence == encoded_text_from_decode
    assert encoded_pair == encoded_pair_from_decode

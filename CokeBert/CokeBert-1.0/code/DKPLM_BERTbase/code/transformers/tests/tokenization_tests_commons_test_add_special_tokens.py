def test_add_special_tokens(self):
    tokenizer = self.get_tokenizer()
    input_text, output_text = self.get_input_output_texts()
    special_token = '[SPECIAL TOKEN]'
    tokenizer.add_special_tokens({'cls_token': special_token})
    encoded_special_token = tokenizer.encode(special_token,
        add_special_tokens=False)
    assert len(encoded_special_token) == 1
    text = ' '.join([input_text, special_token, output_text])
    encoded = tokenizer.encode(text, add_special_tokens=False)
    input_encoded = tokenizer.encode(input_text, add_special_tokens=False)
    output_encoded = tokenizer.encode(output_text, add_special_tokens=False)
    special_token_id = tokenizer.encode(special_token, add_special_tokens=False
        )
    assert encoded == input_encoded + special_token_id + output_encoded
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert special_token not in decoded

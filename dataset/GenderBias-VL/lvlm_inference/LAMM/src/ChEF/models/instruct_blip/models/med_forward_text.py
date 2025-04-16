def forward_text(self, tokenized_text, **kwargs):
    text = tokenized_text
    token_type_ids = kwargs.get('token_type_ids', None)
    text_output = super().forward(text.input_ids, attention_mask=text.
        attention_mask, token_type_ids=token_type_ids, return_dict=True,
        mode='text')
    return text_output

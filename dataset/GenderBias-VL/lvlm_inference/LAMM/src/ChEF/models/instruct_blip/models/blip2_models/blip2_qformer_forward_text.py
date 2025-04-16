def forward_text(self, text_tokens):
    text_output = self.Qformer.bert(text_tokens.input_ids, attention_mask=
        text_tokens.attention_mask, return_dict=True)
    return text_output.last_hidden_state[:, 0, :]

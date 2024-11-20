def moe_set_gate(self, input_texts, device):
    if self.training:
        input_tokens = []
        for input_text in input_texts:
            assert input_text[0]['from'] == 'human'
            token = self.llama_tokenizer(input_text[0]['value'],
                add_special_tokens=False).input_ids
            input_tokens.append(torch.LongTensor(token))
        input_tokens = rnn.pad_sequence(input_tokens, batch_first=True,
            padding_value=self.llama_tokenizer.pad_token_id).to(device)
        input_embeds = self.llama_model.model.model.embed_tokens(input_tokens)
    else:
        input_embeds = input_texts
    soft_gate = self.gating_network(input_embeds, reduce_token=True)
    for _, module in self.llama_model.named_modules():
        if isinstance(module, MoeLoraLayer):
            module.set_gate(soft_gate)
    return

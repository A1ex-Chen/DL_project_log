def forward(self, tokens, labels, imgs):
    visual_query = self.forward_visual(imgs)
    _bsz, seqlen = tokens.shape
    h = self.llama.tok_embeddings(tokens)
    freqs_cis = self.llama.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[:seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=h.device)
    mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
    for layer in self.llama.layers[:-1 * self.query_layer]:
        h = layer(h, 0, freqs_cis, mask)
    adapter = self.adapter_query.weight.reshape(self.query_layer, self.
        query_len, -1).unsqueeze(1)
    adapter_index = 0
    for layer in self.llama.layers[-1 * self.query_layer:]:
        dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
        dynamic_adapter = dynamic_adapter + visual_query
        h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
        adapter_index = adapter_index + 1
    h = self.llama.norm(h)
    output = self.llama.output(h)
    output = output[:, :-1, :]
    labels = labels[:, 1:]
    if labels.sum() == 0:
        c_loss = output.mean() * 0
    else:
        assert self.llama.vocab_size == 32000
        c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size),
            labels.flatten())
    return c_loss, c_loss

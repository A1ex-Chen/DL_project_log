@torch.inference_mode()
def forward_inference(self, visual_query, tokens, start_pos: int, ppl=False):
    _bsz, seqlen = tokens.shape
    h = self.llama.tok_embeddings(tokens)
    freqs_cis = self.llama.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[start_pos:start_pos + seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float('-inf'), device=h.device)
    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
    for layer in self.llama.layers[:-1 * self.query_layer]:
        h = layer(h, start_pos, freqs_cis, mask)
    adapter = self.adapter_query.weight.reshape(self.query_layer, self.
        query_len, -1).unsqueeze(1)
    adapter_index = 0
    for layer in self.llama.layers[-1 * self.query_layer:]:
        dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
        dynamic_adapter = dynamic_adapter + visual_query
        h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
        adapter_index = adapter_index + 1
    h = self.llama.norm(h)
    if ppl:
        return self.llama.output(h).float()
    output = self.llama.output(h[:, -1, :])
    return output.float()

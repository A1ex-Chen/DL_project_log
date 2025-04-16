@torch.jit.export
def forward_decoder(self, tokens, encoder_outs: List[Dict[str, List[Tensor]
    ]], incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
    temperature: float=1.0, multimodal: bool=False):
    log_probs = []
    avg_attn: Optional[Tensor] = None
    encoder_out: Optional[Dict[str, List[Tensor]]] = None
    for i, model in enumerate(self.models):
        if self.has_encoder():
            encoder_out = encoder_outs[i]
        if self.has_incremental_states():
            if hasattr(model, 'gpt_model'):
                decoder_out = model.gpt_model.decoder.forward(tokens,
                    encoder_out=encoder_out, incremental_state=
                    incremental_states[i])
            else:
                decoder_out = model.decoder.forward(tokens, encoder_out=
                    encoder_out, incremental_state=incremental_states[i])
        elif incremental_states is not None and hasattr(model, 'gpt_model'
            ) and multimodal:
            decoder_out = model.gpt_model.decoder.forward(tokens,
                encoder_out=encoder_out, incremental_state=
                incremental_states[i])
        elif hasattr(model, 'decoder'):
            decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out
                )
        else:
            decoder_out = model.forward(tokens)
        attn: Optional[Tensor] = None
        decoder_len = len(decoder_out)
        if decoder_len > 1 and decoder_out[1] is not None:
            if isinstance(decoder_out[1], Tensor):
                attn = decoder_out[1]
            else:
                attn_holder = decoder_out[1]['attn']
                if isinstance(attn_holder, Tensor):
                    attn = attn_holder
                elif attn_holder is not None:
                    attn = attn_holder[0]
            if attn is not None:
                attn = attn[:, -1, :]
        decoder_out_tuple = decoder_out[0][:, -1:, :].div_(temperature
            ), None if decoder_len <= 1 else decoder_out[1]
        probs = model.get_normalized_probs(decoder_out_tuple, log_probs=
            True, sample=None)
        probs = probs[:, -1, :]
        if self.models_size == 1:
            return probs, attn
        log_probs.append(probs)
        if attn is not None:
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
    avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0
        ) - math.log(self.models_size)
    if avg_attn is not None:
        avg_attn.div_(self.models_size)
    return avg_probs, avg_attn

@torch.no_grad()
def generate_beam(self, input_ids=None, input_embeds=None, device=None,
    beam_size: int=5, entry_length: int=67, temperature: float=1.0,
    eos_token_id: Optional[int]=None):
    """
        Generates text using the given tokenizer and text prompt or token embedding via beam search. This
        implementation is based on the beam search implementation from the [original UniDiffuser
        code](https://github.com/thu-ml/unidiffuser/blob/main/libs/caption_decoder.py#L89).

        Args:
            eos_token_id (`int`, *optional*):
                The token ID of the EOS token for the text decoder model.
            input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
                Tokenizer indices of input sequence tokens in the vocabulary. One of `input_ids` and `input_embeds`
                must be supplied.
            input_embeds (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`, *optional*):
                An embedded representation to directly pass to the transformer as a prefix for beam search. One of
                `input_ids` and `input_embeds` must be supplied.
            device:
                The device to perform beam search on.
            beam_size (`int`, *optional*, defaults to `5`):
                The number of best states to store during beam search.
            entry_length (`int`, *optional*, defaults to `67`):
                The number of iterations to run beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The temperature to use when performing the softmax over logits from the decoding model.

        Returns:
            `Tuple(torch.Tensor, torch.Tensor)`: A tuple of tensors where the first element is a tensor of generated
            token sequences sorted by score in descending order, and the second element is the sequence lengths
            corresponding to those sequences.
        """
    stop_token_index = eos_token_id
    tokens = None
    scores = None
    seq_lengths = torch.ones(beam_size, device=device, dtype=torch.int)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    if input_embeds is not None:
        generated = input_embeds
    else:
        generated = self.transformer.transformer.wte(input_ids)
    for i in range(entry_length):
        outputs = self.transformer(inputs_embeds=generated)
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits = logits.softmax(-1).log()
        if scores is None:
            scores, next_tokens = logits.topk(beam_size, -1)
            generated = generated.expand(beam_size, *generated.shape[1:])
            next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = tokens.expand(beam_size, *tokens.shape[1:])
                tokens = torch.cat((tokens, next_tokens), dim=1)
        else:
            logits[is_stopped] = -float(np.inf)
            logits[is_stopped, 0] = 0
            scores_sum = scores[:, None] + logits
            seq_lengths[~is_stopped] += 1
            scores_sum_average = scores_sum / seq_lengths[:, None]
            scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                beam_size, -1)
            next_tokens_source = next_tokens // scores_sum.shape[1]
            seq_lengths = seq_lengths[next_tokens_source]
            next_tokens = next_tokens % scores_sum.shape[1]
            next_tokens = next_tokens.unsqueeze(1)
            tokens = tokens[next_tokens_source]
            tokens = torch.cat((tokens, next_tokens), dim=1)
            generated = generated[next_tokens_source]
            scores = scores_sum_average * seq_lengths
            is_stopped = is_stopped[next_tokens_source]
        next_token_embed = self.transformer.transformer.wte(next_tokens.
            squeeze()).view(generated.shape[0], 1, -1)
        generated = torch.cat((generated, next_token_embed), dim=1)
        is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
        if is_stopped.all():
            break
    scores = scores / seq_lengths
    order = scores.argsort(descending=True)
    output_texts = [tokens[i] for i in order]
    output_texts = torch.stack(output_texts, dim=0)
    seq_lengths = torch.tensor([seq_lengths[i] for i in order], dtype=
        seq_lengths.dtype)
    return output_texts, seq_lengths

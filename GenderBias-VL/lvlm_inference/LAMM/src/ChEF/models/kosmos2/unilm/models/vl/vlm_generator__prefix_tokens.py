def _prefix_tokens(self, step: int, lprobs, scores, tokens, prefix_tokens,
    beam_size: int):
    """Handle prefix tokens"""
    prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size
        ).view(-1)
    prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
    prefix_mask = prefix_toks.ne(self.pad)
    lprobs[prefix_mask] = torch.tensor(-math.inf).to(lprobs)
    lprobs[prefix_mask] = lprobs[prefix_mask].scatter(-1, prefix_toks[
        prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask])
    eos_mask = prefix_toks.eq(self.eos)
    if eos_mask.any():
        first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[
            :, 0, 1:step + 1]
        eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
        target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
        assert (first_beam == target_prefix).all()
        tokens = self.replicate_first_beam(tokens, eos_mask_batch_dim,
            beam_size)
        scores = self.replicate_first_beam(scores, eos_mask_batch_dim,
            beam_size)
        lprobs = self.replicate_first_beam(lprobs, eos_mask_batch_dim,
            beam_size)
    return lprobs, tokens, scores

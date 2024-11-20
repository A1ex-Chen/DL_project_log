def unpad_input(self, hidden_states, attention_mask):
    valid_mask = attention_mask.squeeze(1).squeeze(1).eq(0)
    seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.
        torch.int32), (1, 0))
    hidden_states = hidden_states[indices]
    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

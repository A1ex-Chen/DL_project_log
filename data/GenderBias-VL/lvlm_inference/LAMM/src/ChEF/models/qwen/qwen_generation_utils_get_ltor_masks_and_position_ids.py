def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids,
    reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""
    micro_batch_size, seq_length = data.size()
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length,
        seq_length), device=data.device)).view(att_mask_batch, 1,
        seq_length, seq_length)
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.
        device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if reset_position_ids:
        position_ids = position_ids.clone()
    if reset_position_ids or reset_attention_mask:
        for b in range(micro_batch_size):
            eod_index = position_ids[b, data[b] == eod_token]
            if reset_position_ids:
                eod_index = eod_index.clone()
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                if reset_attention_mask:
                    attention_mask[b, 0, i + 1:, :i + 1] = 0
                if reset_position_ids:
                    position_ids[b, i + 1:] -= i + 1 - prev_index
                    prev_index = i + 1
    attention_mask = attention_mask < 0.5
    return attention_mask, loss_mask, position_ids

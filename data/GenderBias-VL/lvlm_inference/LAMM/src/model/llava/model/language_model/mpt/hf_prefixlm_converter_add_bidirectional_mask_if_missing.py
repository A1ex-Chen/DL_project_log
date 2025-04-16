def add_bidirectional_mask_if_missing(batch: Dict[str, Any]):
    """Attempts to add bidirectional_mask to batch if missing.

    Raises:
        KeyError if bidirectional_mask is missing and can't be inferred
    """
    if 'bidirectional_mask' not in batch:
        if batch.get('mode', None) == 'icl_task':
            batch['bidirectional_mask'] = batch['attention_mask'].clone()
            for i, continuation_indices in enumerate(batch[
                'continuation_indices']):
                batch['bidirectional_mask'][i, continuation_indices] = 0
        elif 'labels' in batch and 'attention_mask' in batch:
            batch['bidirectional_mask'] = torch.logical_and(torch.eq(batch[
                'attention_mask'], 1), torch.eq(batch['labels'], -100)
                ).type_as(batch['attention_mask'])
        else:
            raise KeyError(
                'No bidirectional_mask in batch and not sure how to construct one.'
                )

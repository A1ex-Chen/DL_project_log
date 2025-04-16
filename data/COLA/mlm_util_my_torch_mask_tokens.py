def my_torch_mask_tokens(self, inputs, labels):
    """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
    new_labels = inputs.clone()
    mask_token_indices = inputs == self.tokenizer.mask_token_id
    new_labels[~mask_token_indices] = -100
    new_labels[mask_token_indices] = torch.LongTensor(labels)
    return inputs, new_labels

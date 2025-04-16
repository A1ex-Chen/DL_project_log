def complete_mask(self, has_mask, max_objs, device):
    """
        Based on the input mask corresponding value `0 or 1` for each phrases and image, mask the features
        corresponding to phrases and images.
        """
    mask = torch.ones(1, max_objs).type(self.text_encoder.dtype).to(device)
    if has_mask is None:
        return mask
    if isinstance(has_mask, int):
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0, idx] = value
        return mask

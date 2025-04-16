def create_masks(self, length, num_docs):
    masks = [self.random_mask(length) for _ in range(num_docs)]
    return torch.stack(masks)

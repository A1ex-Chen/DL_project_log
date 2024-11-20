def _compute_block_mask(self, mask):
    block_mask = F.max_pool2d(input=mask[:, None, :, :], kernel_size=(self.
        block_size, self.block_size), stride=(1, 1), padding=self.
        block_size // 2)
    if self.block_size % 2 == 0:
        block_mask = block_mask[:, :, :-1, :-1]
    block_mask = 1 - block_mask.squeeze(1)
    return block_mask

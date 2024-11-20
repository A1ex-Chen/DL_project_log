def stride_pool_pos(self, pos_id, block_index):
    """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
    if self.config.separate_cls:
        cls_pos = pos_id.new_tensor([-2 ** block_index + 1])
        pooled_pos_id = pos_id[1:-1] if self.config.truncate_seq else pos_id[1:
            ]
        return torch.cat([cls_pos, pooled_pos_id[::2]], 0)
    else:
        return pos_id[::2]

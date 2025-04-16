def set_mask(self, context_len, context):
    """
        sets self.mask which is applied before softmax
        ones for inactive context fields, zeros for active context fields

        :param context_len: b
        :param context: if batch_first: (b x t_k x n) else: (t_k x b x n)

        self.mask: (b x t_k)
        """
    if self.batch_first:
        max_len = context.size(1)
    else:
        max_len = context.size(0)
    indices = torch.arange(0, max_len, dtype=torch.int64, device=context.device
        )
    self.mask = indices >= context_len.unsqueeze(1)

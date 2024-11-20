def transpose_for_scores(self, x):
    """
        - input: [N, C, W]
        - output: [N, C1, W, C2] where C1 is the head index, and C2 is one head's contents
        """
    new_x_shape = x.size()[0
        ], self.num_attention_heads, self.attention_head_size, x.size()[-1]
    x = x.view(*new_x_shape)
    return x.permute(0, 1, 3, 2)

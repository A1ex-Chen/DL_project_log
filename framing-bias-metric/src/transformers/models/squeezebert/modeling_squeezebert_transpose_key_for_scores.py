def transpose_key_for_scores(self, x):
    """
        - input: [N, C, W]
        - output: [N, C1, C2, W] where C1 is the head index, and C2 is one head's contents
        """
    new_x_shape = x.size()[0
        ], self.num_attention_heads, self.attention_head_size, x.size()[-1]
    x = x.view(*new_x_shape)
    return x

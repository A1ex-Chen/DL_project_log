def transpose_output(self, x):
    """
        - input: [N, C1, W, C2]
        - output: [N, C, W]
        """
    x = x.permute(0, 1, 3, 2).contiguous()
    new_x_shape = x.size()[0], self.all_head_size, x.size()[3]
    x = x.view(*new_x_shape)
    return x

def unshape(states):
    """  reshape """
    return states.transpose(1, 2).contiguous().view(batch_size, -1, self.
        inner_dim)

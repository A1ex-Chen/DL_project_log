def _parse_hidden_state(self, hx):
    """
        Dealing w. hidden state:
        Typically in pytorch: (h_0, c_0)
            h_0 = ``[num_layers * num_directions, batch, hidden_size]``
            c_0 = ``[num_layers * num_directions, batch, hidden_size]``
        """
    if hx is None:
        return [None] * self.num_layers
    else:
        h_0, c_0 = hx
        assert h_0.shape[0] == self.num_layers
        return [(h_0[i].unsqueeze(dim=0), c_0[i].unsqueeze(dim=0)) for i in
            range(h_0.shape[0])]

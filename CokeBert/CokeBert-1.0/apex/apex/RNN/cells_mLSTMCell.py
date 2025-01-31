def mLSTMCell(input, hidden, w_ih, w_hh, w_mih, w_mhh, b_ih=None, b_hh=None):
    """
    mLSTMCell
    """
    if input.is_cuda:
        igates = F.linear(input, w_ih)
        m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
        hgates = F.linear(m, w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1], b_ih, b_hh)
    hx, cx = hidden
    m = F.linear(input, w_mih) * F.linear(hidden[0], w_mhh)
    gates = F.linear(input, w_ih, b_ih) + F.linear(m, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * F.tanh(cy)
    return hy, cy

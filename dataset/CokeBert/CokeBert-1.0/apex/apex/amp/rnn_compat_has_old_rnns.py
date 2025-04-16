def has_old_rnns():
    try:
        torch.nn.backends.thnn.backend.LSTMCell
        return True
    except:
        return False

def b_inv(b_mat):
    """ Performs batch matrix inversion.

    Arguments:
        b_mat: the batch of matrices that should be inverted
    """
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

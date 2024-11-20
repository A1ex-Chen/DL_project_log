def return_interpolate(self, pred_f, pred_b):
    """ Returns velocity field-based interpolation for forward and
        backward prediction.

        Args:
            pred_f (tensor): forward prediction for vertices
            pred_b (tensor): backward prediction for vertices
        """
    assert pred_f.shape[0] == pred_b.shape[0]
    n_steps = pred_f.shape[0] + 2
    w = np.arange(1, n_steps - 1) / (n_steps - 1)
    w = w[:, np.newaxis, np.newaxis]
    pred_out = pred_f * (1 - w) + pred_b * w
    return pred_out

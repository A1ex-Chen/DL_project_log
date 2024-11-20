def __init__(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True,
    use_vfl=False, use_uni_match=False, uni_match_ind=0):
    """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
    super().__init__()
    if loss_gain is None:
        loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1,
            'mask': 1, 'dice': 1}
    self.nc = nc
    self.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5,
        'giou': 2})
    self.loss_gain = loss_gain
    self.aux_loss = aux_loss
    self.fl = FocalLoss() if use_fl else None
    self.vfl = VarifocalLoss() if use_vfl else None
    self.use_uni_match = use_uni_match
    self.uni_match_ind = uni_match_ind
    self.device = None

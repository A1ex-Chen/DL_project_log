def __init__(self, model, autobalance=False):
    device = next(model.parameters()).device
    h = model.hyp
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']],
        device=device))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']],
        device=device))
    self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
    g = h['fl_gamma']
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    m = de_parallel(model).model[-1]
    self.balance = {(3): [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 
        0.02])
    self.ssi = list(m.stride).index(16) if autobalance else 0
    self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (BCEcls,
        BCEobj, 1.0, h, autobalance)
    self.na = m.na
    self.nc = m.nc
    self.nl = m.nl
    self.anchors = m.anchors
    self.device = device

def _forward_once(self, x):
    out = self.backbone(x)
    for neck in self.necks:
        out = neck(out)
    y = self.detection(list(out))
    return y

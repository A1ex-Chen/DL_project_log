def __call__(self, p, targets):
    targets = targets.to(self.device)
    if not self.det.training or len(p) == 0 or len(targets) == 0:
        return torch.tensor(0.0, device=self.device, requires_grad=True
            ), torch.zeros(4, device=self.device, requires_grad=True)
    loss, iou_loss, obj_loss, cls_loss, l1_loss, _ = self.det.get_losses(*p,
        targets, dtype=p[0].dtype)
    return loss, torch.hstack((iou_loss, obj_loss, cls_loss, l1_loss)).detach()

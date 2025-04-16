def losses(self, predictions, targets):
    predictions = predictions.float()
    predictions = F.interpolate(predictions, scale_factor=self.
        common_stride, mode='bilinear', align_corners=False)
    loss = F.cross_entropy(predictions, targets, reduction='mean',
        ignore_index=self.ignore_value)
    losses = {'loss_sem_seg': loss * self.loss_weight}
    return losses

def loss(self, logit_batch: Tensor, gt_classes_batch: Tensor) ->Tensor:
    loss_batch = F.cross_entropy(input=logit_batch, target=gt_classes_batch,
        reduction='none')
    return loss_batch

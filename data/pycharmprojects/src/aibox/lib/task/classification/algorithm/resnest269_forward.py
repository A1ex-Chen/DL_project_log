def forward(self, padded_image_batch: Tensor, gt_classes_batch: Tensor=None
    ) ->Union[Tensor, Tuple[Tensor, Tensor]]:
    batch_size, _, padded_image_height, padded_image_width = (
        padded_image_batch.shape)
    logit_batch = self.net.forward(padded_image_batch)
    if self.training:
        loss_batch = self.loss(logit_batch, gt_classes_batch)
        return loss_batch
    else:
        pred_prob_batch, pred_class_batch = F.softmax(input=logit_batch, dim=1
            ).max(dim=1)
        return pred_prob_batch, pred_class_batch

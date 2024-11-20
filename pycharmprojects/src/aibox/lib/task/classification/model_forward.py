def forward(self, image_batch: Bunch, gt_classes_batch: Bunch=None) ->Union[
    Bunch, Tuple[Bunch, Bunch]]:
    batch_size = len(image_batch)
    if batch_size == 1:
        padded_image_batch = image_batch[0].unsqueeze(dim=0)
    else:
        padded_image_width = max([it.shape[2] for it in image_batch])
        padded_image_height = max([it.shape[1] for it in image_batch])
        padded_image_batch = []
        for image in image_batch:
            padded_image = F.pad(input=image, pad=[0, padded_image_width -
                image.shape[2], 0, padded_image_height - image.shape[1]])
            padded_image_batch.append(padded_image)
        padded_image_batch = torch.stack(padded_image_batch, dim=0)
    padded_image_batch = normalize_means_stds(padded_image_batch, list(self
        .algorithm.normalization_means()), list(self.algorithm.
        normalization_stds()))
    if self.training:
        gt_classes_batch = torch.stack(gt_classes_batch, dim=0)
        loss_batch = self.algorithm.forward(padded_image_batch,
            gt_classes_batch)
        loss_batch = Bunch(loss_batch.unbind(dim=0))
        return loss_batch
    else:
        pred_prob_batch, pred_class_batch = self.algorithm.forward(
            padded_image_batch)
        pred_prob_batch = Bunch(pred_prob_batch.unbind(dim=0))
        pred_class_batch = Bunch(pred_class_batch.unbind(dim=0))
        return pred_prob_batch, pred_class_batch

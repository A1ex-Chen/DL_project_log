def forward(self, inputs_scales, targets_scales):
    losses = []
    for inputs, targets in zip(inputs_scales, targets_scales):
        targets_m = targets.clone()
        targets_m -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        number_of_pixels_per_class = torch.bincount(targets.flatten().type(
            self.dtype), minlength=self.num_classes)
        divisor_weighted_pixel_sum = torch.sum(number_of_pixels_per_class[1
            :] * self.weight)
        losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
    return losses

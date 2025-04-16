def get_rgb_full_loss(self, rgb_values, rgb_gt, rgb_loss_type='l2'):
    if rgb_loss_type == 'l1':
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1]
            )
    elif rgb_loss_type == 'l2':
        rgb_loss = self.l2_loss(rgb_values, rgb_gt) / float(rgb_values.shape[1]
            )
    return rgb_loss

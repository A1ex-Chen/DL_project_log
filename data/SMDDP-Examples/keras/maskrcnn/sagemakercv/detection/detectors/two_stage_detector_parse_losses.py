def parse_losses(self, losses, weight_decay=0.0):
    loss_dict = {i: j for i, j in losses.items() if 'loss' in i and 'total'
         not in i}
    if weight_decay > 0.0:
        loss_dict['l2_loss'] = weight_decay * tf.add_n([tf.nn.l2_loss(v) for
            v in self.trainable_variables if not any([(pattern in v.name) for
            pattern in ['batch_normalization', 'bias', 'beta']])])
    loss_dict['total_loss'] = sum(loss_dict.values())
    return loss_dict

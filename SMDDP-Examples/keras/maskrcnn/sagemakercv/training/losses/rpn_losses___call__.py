def __call__(self, score_outputs, box_outputs, labels):
    with tf.name_scope('rpn_loss'):
        score_losses = []
        box_losses = []
        for level in range(int(self.min_level), int(self.max_level + 1)):
            score_targets_at_level = labels['score_targets_%d' % level]
            box_targets_at_level = labels['box_targets_%d' % level]
            score_losses.append(self._rpn_score_loss(score_outputs=
                score_outputs[level], score_targets=score_targets_at_level,
                normalizer=tf.cast(self.train_batch_size_per_gpu * self.
                rpn_batch_size_per_im, dtype=tf.float32)))
            box_losses.append(self._rpn_box_loss(box_outputs=box_outputs[
                level], box_targets=box_targets_at_level, normalizer=1.0))
        rpn_score_loss = tf.add_n(score_losses)
        rpn_box_loss = self.rpn_box_loss_weight * tf.add_n(box_losses)
        total_rpn_loss = rpn_score_loss + rpn_box_loss
    return total_rpn_loss, rpn_score_loss, rpn_box_loss

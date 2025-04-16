@property
def dummy_inputs(self):
    """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
    batch_size = 2
    num_visual_features = 10
    input_ids = tf.constant([[3, 5, 6], [2, 3, 4]])
    visual_feats = tf.random.uniform((batch_size, num_visual_features, self
        .config.visual_feat_dim))
    visual_pos = tf.random.uniform((batch_size, num_visual_features, 4))
    if self.config.task_obj_predict:
        obj_labels = {}
    if self.config.visual_attr_loss and self.config.task_obj_predict:
        obj_labels['attr'] = tf.ones([batch_size, num_visual_features]
            ), tf.ones([batch_size, num_visual_features])
    if self.config.visual_feat_loss and self.config.task_obj_predict:
        obj_labels['feat'] = tf.ones([batch_size, num_visual_features, self
            .config.visual_feat_dim]), tf.ones([batch_size,
            num_visual_features])
    if self.config.visual_obj_loss and self.config.task_obj_predict:
        obj_labels['obj'] = tf.ones([batch_size, num_visual_features]
            ), tf.ones([batch_size, num_visual_features])
    return {**{'input_ids': input_ids, 'visual_feats': visual_feats,
        'visual_pos': visual_pos}, **{'obj_labels': obj_labels} if self.
        config.task_obj_predict else {}}

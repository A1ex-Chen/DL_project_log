def __init__(self, num_classes=91, mlp_head_dim=1024, name='box_head',
    trainable=True, class_agnostic_box=False, loss_cfg=dict(num_classes=91,
    box_loss_type='huber', use_carl_loss=False, bbox_reg_weights=(10.0, 
    10.0, 5.0, 5.0), fast_rcnn_box_loss_weight=1.0, image_size=(832.0, 
    1344.0), class_agnostic_box=False), *args, **kwargs):
    """Box and class branches for the Mask-RCNN model.
        Args:
        roi_features: A ROI feature tensor of shape
          [batch_size, num_rois, height_l, width_l, num_filters].
        num_classes: a integer for the number of classes.
        mlp_head_dim: a integer that is the hidden dimension in the fully-connected
          layers.
        """
    super(StandardBBoxHead, self).__init__(*args, name=name, trainable=
        trainable, **kwargs)
    self._num_classes = num_classes
    self._mlp_head_dim = mlp_head_dim
    self._class_agnostic_box = class_agnostic_box
    self._bbox_dense_0 = tf.keras.layers.Dense(units=mlp_head_dim,
        activation=tf.nn.relu, trainable=trainable, name='bbox_dense_0')
    self._bbox_dense_1 = tf.keras.layers.Dense(units=mlp_head_dim,
        activation=tf.nn.relu, trainable=trainable, name='bbox_dense_1')
    self._dense_class = tf.keras.layers.Dense(num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        bias_initializer=tf.keras.initializers.Zeros(), trainable=trainable,
        name='class-predict')
    self._dense_box = tf.keras.layers.Dense(8 if self._class_agnostic_box else
        num_classes * 4, kernel_initializer=tf.random_normal_initializer(
        stddev=0.001), bias_initializer=tf.keras.initializers.Zeros(),
        trainable=trainable, name='box-predict')
    self.loss = FastRCNNLoss(**loss_cfg)

def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    self.config = config
    self.num_qa_labels = config.num_qa_labels
    self.visual_loss_normalizer = config.visual_loss_normalizer
    self.task_mask_lm = config.task_mask_lm
    self.task_obj_predict = config.task_obj_predict
    self.task_matched = config.task_matched
    self.task_qa = config.task_qa
    self.lxmert = TFLxmertMainLayer(config, name='lxmert')
    self.cls = TFLxmertPreTrainingHeads(config, self.lxmert.embeddings,
        name='cls')
    if self.task_obj_predict:
        self.obj_predict_head = TFLxmertVisualObjHead(config, name=
            'obj_predict_head')
    if self.task_qa:
        self.answer_head = TFLxmertVisualAnswerHead(config, self.
            num_qa_labels, name='answer_head')
    self.loss_fcts = {'l2': tf.keras.losses.Huber(delta=1.0, name=
        'huber_loss'), 'visn_ce': tf.keras.losses.
        SparseCategoricalCrossentropy(from_logits=True), 'ce': tf.keras.
        losses.SparseCategoricalCrossentropy(from_logits=True)}
    visual_losses = {}
    if config.visual_obj_loss:
        visual_losses['obj'] = {'shape': (-1,), 'num': config.
            num_object_labels, 'loss': 'visn_ce'}
    if config.visual_attr_loss:
        visual_losses['attr'] = {'shape': (-1,), 'num': config.
            num_attr_labels, 'loss': 'visn_ce'}
    if config.visual_obj_loss:
        visual_losses['feat'] = {'shape': (-1, config.visual_feat_dim),
            'num': config.visual_feat_dim, 'loss': 'l2'}
    self.visual_losses = visual_losses

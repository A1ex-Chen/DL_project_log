def __init__(self, vocab_size=30522, hidden_size=768, num_attention_heads=
    12, num_labels=2, num_qa_labels=9500, num_object_labels=1600,
    num_attr_labels=400, intermediate_size=3072, hidden_act='gelu',
    hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
    layer_norm_eps=1e-12, pad_token_id=0, l_layers=9, x_layers=5, r_layers=
    5, visual_feat_dim=2048, visual_pos_dim=4, visual_loss_normalizer=6.67,
    task_matched=True, task_mask_lm=True, task_obj_predict=True, task_qa=
    True, visual_obj_loss=True, visual_attr_loss=True, visual_feat_loss=
    True, output_attentions=False, output_hidden_states=False, **kwargs):
    super().__init__(**kwargs)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.num_labels = num_labels
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.num_qa_labels = num_qa_labels
    self.num_object_labels = num_object_labels
    self.num_attr_labels = num_attr_labels
    self.l_layers = l_layers
    self.x_layers = x_layers
    self.r_layers = r_layers
    self.visual_feat_dim = visual_feat_dim
    self.visual_pos_dim = visual_pos_dim
    self.visual_loss_normalizer = visual_loss_normalizer
    self.task_matched = task_matched
    self.task_mask_lm = task_mask_lm
    self.task_obj_predict = task_obj_predict
    self.task_qa = task_qa
    self.visual_obj_loss = visual_obj_loss
    self.visual_attr_loss = visual_attr_loss
    self.visual_feat_loss = visual_feat_loss
    self.output_hidden_states = output_hidden_states
    self.output_attentions = self.output_attentions
    self.num_hidden_layers = {'vision': r_layers, 'cross_encoder': x_layers,
        'language': l_layers}

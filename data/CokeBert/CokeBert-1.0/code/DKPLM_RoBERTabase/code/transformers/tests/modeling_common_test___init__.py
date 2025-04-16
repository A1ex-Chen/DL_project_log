def __init__(self, parent, batch_size=13, seq_length=7, is_training=True,
    use_position_ids=True, use_token_type_ids=True, use_labels=True,
    vocab_size=99, n_positions=33, hidden_size=32, num_hidden_layers=5,
    num_attention_heads=4, n_choices=3, type_sequence_label_size=2,
    initializer_range=0.02, num_labels=3, scope=None, config_class=None,
    base_model_class=None, lm_head_model_class=None,
    double_head_model_class=None):
    self.parent = parent
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.is_training = is_training
    self.use_position_ids = use_position_ids
    self.use_token_type_ids = use_token_type_ids
    self.use_labels = use_labels
    self.vocab_size = vocab_size
    self.n_positions = n_positions
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.n_choices = n_choices
    self.type_sequence_label_size = type_sequence_label_size
    self.initializer_range = initializer_range
    self.num_labels = num_labels
    self.scope = scope
    self.config_class = config_class
    self.base_model_class = base_model_class
    self.lm_head_model_class = lm_head_model_class
    self.double_head_model_class = double_head_model_class
    self.all_model_classes = (base_model_class, lm_head_model_class,
        double_head_model_class)

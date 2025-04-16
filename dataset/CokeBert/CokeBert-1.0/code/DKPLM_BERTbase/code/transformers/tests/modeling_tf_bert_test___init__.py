def __init__(self, parent, batch_size=13, seq_length=7, is_training=True,
    use_input_mask=True, use_token_type_ids=True, use_labels=True,
    vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads
    =4, intermediate_size=37, hidden_act='gelu', hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1, max_position_embeddings=512,
    type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02,
    num_labels=3, num_choices=4, scope=None):
    self.parent = parent
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.is_training = is_training
    self.use_input_mask = use_input_mask
    self.use_token_type_ids = use_token_type_ids
    self.use_labels = use_labels
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.hidden_act = hidden_act
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.type_sequence_label_size = type_sequence_label_size
    self.initializer_range = initializer_range
    self.num_labels = num_labels
    self.num_choices = num_choices
    self.scope = scope

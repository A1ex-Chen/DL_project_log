def __init__(self, parent, batch_size=13, seq_length=7, is_training=True,
    use_input_lengths=True, use_token_type_ids=True, use_labels=True,
    gelu_activation=True, sinusoidal_embeddings=False, causal=False, asm=
    False, n_langs=2, vocab_size=99, n_special=0, hidden_size=32,
    num_hidden_layers=5, num_attention_heads=4, hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1, max_position_embeddings=512,
    type_vocab_size=16, type_sequence_label_size=2, initializer_range=0.02,
    num_labels=3, num_choices=4, summary_type='last', use_proj=True, scope=None
    ):
    self.parent = parent
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.is_training = is_training
    self.use_input_lengths = use_input_lengths
    self.use_token_type_ids = use_token_type_ids
    self.use_labels = use_labels
    self.gelu_activation = gelu_activation
    self.sinusoidal_embeddings = sinusoidal_embeddings
    self.asm = asm
    self.n_langs = n_langs
    self.vocab_size = vocab_size
    self.n_special = n_special
    self.summary_type = summary_type
    self.causal = causal
    self.use_proj = use_proj
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.n_langs = n_langs
    self.type_sequence_label_size = type_sequence_label_size
    self.initializer_range = initializer_range
    self.summary_type = summary_type
    self.num_labels = num_labels
    self.num_choices = num_choices
    self.scope = scope

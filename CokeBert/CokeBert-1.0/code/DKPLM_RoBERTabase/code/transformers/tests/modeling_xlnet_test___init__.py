def __init__(self, parent, batch_size=13, seq_length=7, mem_len=10,
    clamp_len=-1, reuse_len=15, is_training=True, use_labels=True,
    vocab_size=99, cutoffs=[10, 50, 80], hidden_size=32,
    num_attention_heads=4, d_inner=128, num_hidden_layers=5,
    max_position_embeddings=10, type_sequence_label_size=2, untie_r=True,
    bi_data=False, same_length=False, initializer_range=0.05, seed=1,
    type_vocab_size=2):
    self.parent = parent
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.mem_len = mem_len
    self.clamp_len = clamp_len
    self.reuse_len = reuse_len
    self.is_training = is_training
    self.use_labels = use_labels
    self.vocab_size = vocab_size
    self.cutoffs = cutoffs
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.d_inner = d_inner
    self.num_hidden_layers = num_hidden_layers
    self.max_position_embeddings = max_position_embeddings
    self.bi_data = bi_data
    self.untie_r = untie_r
    self.same_length = same_length
    self.initializer_range = initializer_range
    self.seed = seed
    self.type_vocab_size = type_vocab_size
    self.type_sequence_label_size = type_sequence_label_size

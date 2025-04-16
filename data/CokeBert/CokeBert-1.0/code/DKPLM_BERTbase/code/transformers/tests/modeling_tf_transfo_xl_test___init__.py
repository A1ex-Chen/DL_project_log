def __init__(self, parent, batch_size=13, seq_length=7, mem_len=30,
    clamp_len=15, is_training=True, use_labels=True, vocab_size=99, cutoffs
    =[10, 50, 80], hidden_size=32, d_embed=32, num_attention_heads=4,
    d_head=8, d_inner=128, div_val=2, num_hidden_layers=5, scope=None, seed=1):
    self.parent = parent
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.mem_len = mem_len
    self.key_len = seq_length + mem_len
    self.clamp_len = clamp_len
    self.is_training = is_training
    self.use_labels = use_labels
    self.vocab_size = vocab_size
    self.cutoffs = cutoffs
    self.hidden_size = hidden_size
    self.d_embed = d_embed
    self.num_attention_heads = num_attention_heads
    self.d_head = d_head
    self.d_inner = d_inner
    self.div_val = div_val
    self.num_hidden_layers = num_hidden_layers
    self.scope = scope
    self.seed = seed

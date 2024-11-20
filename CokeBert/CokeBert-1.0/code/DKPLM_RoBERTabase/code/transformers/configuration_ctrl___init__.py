def __init__(self, vocab_size_or_config_json_file=246534, n_positions=256,
    n_ctx=256, n_embd=1280, dff=8192, n_layer=48, n_head=16, resid_pdrop=
    0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-06,
    initializer_range=0.02, num_labels=1, summary_type='cls_index',
    summary_use_proj=True, summary_activation=None, summary_proj_to_labels=
    True, summary_first_dropout=0.1, **kwargs):
    """Constructs CTRLConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CTRLModel` or a configuration json file.
            n_positions: Number of positional embeddings.
            n_ctx: Size of the causal mask (usually same as n_positions).
            dff: Size of the inner dimension of the FFN.
            n_embd: Dimensionality of the embeddings and hidden states.
            n_layer: Number of hidden layers in the Transformer encoder.
            n_head: Number of attention heads for each attention layer in
                the Transformer encoder.
            layer_norm_epsilon: epsilon to use in the layer norm layers
            resid_pdrop: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attn_pdrop: The dropout ratio for the attention
                probabilities.
            embd_pdrop: The dropout ratio for the embeddings.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
    super(CTRLConfig, self).__init__(**kwargs)
    self.vocab_size = vocab_size_or_config_json_file if isinstance(
        vocab_size_or_config_json_file, int) else -1
    self.n_ctx = n_ctx
    self.n_positions = n_positions
    self.n_embd = n_embd
    self.n_layer = n_layer
    self.n_head = n_head
    self.dff = dff
    self.resid_pdrop = resid_pdrop
    self.embd_pdrop = embd_pdrop
    self.attn_pdrop = attn_pdrop
    self.layer_norm_epsilon = layer_norm_epsilon
    self.initializer_range = initializer_range
    self.num_labels = num_labels
    self.summary_type = summary_type
    self.summary_use_proj = summary_use_proj
    self.summary_activation = summary_activation
    self.summary_first_dropout = summary_first_dropout
    self.summary_proj_to_labels = summary_proj_to_labels
    if isinstance(vocab_size_or_config_json_file, str) or sys.version_info[0
        ] == 2 and isinstance(vocab_size_or_config_json_file, unicode):
        with open(vocab_size_or_config_json_file, 'r', encoding='utf-8'
            ) as reader:
            json_config = json.loads(reader.read())
        for key, value in json_config.items():
            self.__dict__[key] = value
    elif not isinstance(vocab_size_or_config_json_file, int):
        raise ValueError(
            'First argument must be either a vocabulary size (int)or the path to a pretrained model config file (str)'
            )

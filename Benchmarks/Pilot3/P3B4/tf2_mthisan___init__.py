def __init__(self, embedding_matrix, num_classes, attention_size,
    attention_heads):
    super(mthisan.mthisan_model, self).__init__()
    self.attention_size = attention_size
    self.attention_heads = attention_heads
    self.training = False
    self.embedding = layers.Embedding(embedding_matrix.shape[0],
        embedding_matrix.shape[1], embeddings_initializer=tf.keras.
        initializers.Constant(embedding_matrix.astype(np.float32)),
        trainable=False)
    self.word_drop = layers.Dropout(0.1)
    self.word_Q = layers.Dense(self.attention_size)
    self.word_K = layers.Dense(self.attention_size)
    self.word_V = layers.Dense(self.attention_size)
    self.word_target = tf.Variable(tf.random.uniform(shape=[1, self.
        attention_heads, 1, int(self.attention_size / self.attention_heads)]))
    self.word_self_att = scaled_attention(use_scale=1 / np.sqrt(
        attention_size), dropout=0.1)
    self.word_targ_att = scaled_attention(use_scale=1 / np.sqrt(
        attention_size), dropout=0.1)
    self.line_drop = layers.Dropout(0.1)
    self.line_Q = layers.Dense(self.attention_size)
    self.line_K = layers.Dense(self.attention_size)
    self.line_V = layers.Dense(self.attention_size)
    self.line_target = tf.Variable(tf.random.uniform(shape=[1, self.
        attention_heads, 1, int(self.attention_size / self.attention_heads)]))
    self.line_self_att = scaled_attention(use_scale=1 / np.sqrt(
        attention_size), dropout=0.1)
    self.line_targ_att = scaled_attention(use_scale=1 / np.sqrt(
        attention_size), dropout=0.1)
    self.doc_drop = layers.Dropout(0.1)
    self.classify_layers = []
    for c in num_classes:
        self.classify_layers.append(layers.Dense(c))

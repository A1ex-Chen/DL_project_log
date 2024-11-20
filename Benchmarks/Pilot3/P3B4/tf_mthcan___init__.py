def __init__(self, embedding_matrix, num_classes, max_sents, max_words,
    attention_size=512, dropout_rate=0.9, activation=tf.nn.elu, lr=0.0001,
    optimizer='adam', embed_train=True):
    tf.compat.v1.reset_default_graph()
    dropout_keep = dropout_rate
    self.dropout_keep = dropout_keep
    self.dropout = tf.compat.v1.placeholder(tf.float32)
    self.ms = max_sents
    self.mw = max_words
    self.embedding_matrix = embedding_matrix.astype(np.float32)
    self.attention_size = attention_size
    self.activation = activation
    self.num_tasks = len(num_classes)
    self.embed_train = embed_train
    self.doc_input = tf.compat.v1.placeholder(tf.int32, shape=[None,
        max_sents, max_words])
    batch_size = tf.shape(self.doc_input)[0]
    words_per_sent = tf.reduce_sum(tf.sign(self.doc_input), 2)
    max_words_ = tf.reduce_max(words_per_sent)
    sents_per_doc = tf.reduce_sum(tf.sign(words_per_sent), 1)
    max_sents_ = tf.reduce_max(sents_per_doc)
    doc_input_reduced = self.doc_input[:, :max_sents_, :max_words_]
    doc_input_reshape = tf.reshape(doc_input_reduced, (-1, max_words_))
    word_embeds = tf.gather(tf.compat.v1.get_variable('embeddings',
        initializer=self.embedding_matrix, dtype=tf.float32, trainable=self
        .embed_train), doc_input_reshape)
    word_embeds = tf.nn.dropout(word_embeds, self.dropout)
    Q = tf.compat.v1.layers.conv1d(word_embeds, self.attention_size, 1,
        padding='same', activation=self.activation, kernel_initializer=tf.
        initializers.glorot_uniform())
    K = tf.compat.v1.layers.conv1d(word_embeds, self.attention_size, 1,
        padding='same', activation=self.activation, kernel_initializer=tf.
        initializers.glorot_uniform())
    V = tf.compat.v1.layers.conv1d(word_embeds, self.attention_size, 1,
        padding='same', activation=self.activation, kernel_initializer=tf.
        initializers.glorot_uniform())
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    outputs = outputs / K.get_shape().as_list()[-1] ** 0.5
    outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs) * -1000,
        outputs)
    outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
    outputs = tf.matmul(outputs, V)
    Q = tf.compat.v1.get_variable('word_Q', (1, 1, self.attention_size), tf
        .float32, tf.initializers.orthogonal())
    Q = tf.tile(Q, [batch_size * max_sents_, 1, 1])
    V = outputs
    outputs = tf.matmul(Q, tf.transpose(outputs, [0, 2, 1]))
    outputs = outputs / K.get_shape().as_list()[-1] ** 0.5
    outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs) * -1000,
        outputs)
    outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
    outputs = tf.matmul(outputs, V)
    sent_embeds = tf.reshape(outputs, (-1, max_sents_, self.attention_size))
    sent_embeds = tf.nn.dropout(sent_embeds, self.dropout)
    Q = tf.compat.v1.layers.conv1d(sent_embeds, self.attention_size, 1,
        padding='same', activation=self.activation, kernel_initializer=tf.
        initializers.glorot_uniform())
    K = tf.compat.v1.layers.conv1d(sent_embeds, self.attention_size, 1,
        padding='same', activation=self.activation, kernel_initializer=tf.
        initializers.glorot_uniform())
    V = tf.compat.v1.layers.conv1d(sent_embeds, self.attention_size, 1,
        padding='same', activation=self.activation, kernel_initializer=tf.
        initializers.glorot_uniform())
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
    outputs = outputs / K.get_shape().as_list()[-1] ** 0.5
    outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs) * -1000,
        outputs)
    outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
    outputs = tf.matmul(outputs, V)
    Q = tf.compat.v1.get_variable('sent_Q', (1, 1, self.attention_size), tf
        .float32, tf.initializers.orthogonal())
    Q = tf.tile(Q, [batch_size, 1, 1])
    V = outputs
    outputs = tf.matmul(Q, tf.transpose(outputs, [0, 2, 1]))
    outputs = outputs / K.get_shape().as_list()[-1] ** 0.5
    outputs = tf.where(tf.equal(outputs, 0), tf.ones_like(outputs) * -1000,
        outputs)
    outputs = tf.nn.dropout(tf.nn.softmax(outputs), self.dropout)
    outputs = tf.matmul(outputs, V)
    doc_embeds = tf.nn.dropout(tf.squeeze(outputs, [1]), self.dropout)
    logits = []
    self.predictions = []
    for i in range(self.num_tasks):
        logit = tf.compat.v1.layers.dense(doc_embeds, num_classes[i],
            kernel_initializer=tf.initializers.glorot_uniform())
        logits.append(logit)
        self.predictions.append(tf.nn.softmax(logit))
    self.labels = []
    self.loss = 0
    for i in range(self.num_tasks):
        label = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.labels.append(label)
        loss = tf.reduce_mean(tf.nn.
            sparse_softmax_cross_entropy_with_logits(logits=logits[i],
            labels=label))
        self.loss += loss / self.num_tasks
    if optimizer == 'adam':
        self.optimizer = tf.compat.v1.train.AdamOptimizer(lr, 0.9, 0.99)
    elif optimizer == 'sgd':
        self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
    elif optimizer == 'adadelta':
        self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate=lr)
    else:
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
    tf_version = tf.__version__
    tf_version_split = tf_version.split('.')
    if int(tf_version_split[0]) == 1 and int(tf_version_split[1]) > 13:
        self.optimizer = (tf.train.experimental.
            enable_mixed_precision_graph_rewrite(self.optimizer, loss_scale
            ='dynamic'))
    self.optimizer = self.optimizer.minimize(self.loss)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.saver = tf.compat.v1.train.Saver()
    self.sess = tf.compat.v1.Session(config=config)
    self.sess.run(tf.compat.v1.global_variables_initializer())

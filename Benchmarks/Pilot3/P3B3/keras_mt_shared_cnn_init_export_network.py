def init_export_network(task_names, task_list, num_classes, in_seq_len,
    vocab_size, wv_space, filter_sizes, num_filters, concat_dropout_prob,
    emb_l2, w_l2, optimizer):
    input_shape = tuple([in_seq_len])
    model_input = Input(shape=input_shape, name='Input')
    emb_lookup = Embedding(vocab_size, wv_space, input_length=in_seq_len,
        name='embedding', embeddings_regularizer=l2(emb_l2))(model_input)
    conv_blocks = []
    for ith_filter, sz in enumerate(filter_sizes):
        conv = Convolution1D(filters=num_filters[ith_filter], kernel_size=
            sz, padding='same', activation='relu', strides=1, name=str(
            ith_filter) + '_thfilter')(emb_lookup)
        conv_blocks.append(GlobalMaxPooling1D()(conv))
    concat = Concatenate()(conv_blocks) if len(conv_blocks
        ) > 1 else conv_blocks[0]
    concat_drop = Dropout(concat_dropout_prob)(concat)
    FC_models = []
    for i in range(len(task_names)):
        if i in task_list:
            outlayer = Dense(num_classes[i], name=task_names[i], activation
                ='softmax')(concat_drop)
            FC_models.append(outlayer)
    model = Model(inputs=model_input, outputs=FC_models)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=
        optimizer, metrics=['acc'])
    return model

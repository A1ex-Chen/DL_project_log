def call(self, inputs, training=False):
    """ hidden_states: float Tensor in shape [bsz, seq_len, hidden_size], the hidden-states of the last layer.
            cls_index: [optional] position of the classification token if summary_type == 'cls_index',
                shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
                if summary_type == 'cls_index' and cls_index is None:
                    we take the last token of the sequence as classification token
        """
    if not isinstance(inputs, (dict, tuple, list)):
        hidden_states = inputs
        cls_index = None
    elif isinstance(inputs, (tuple, list)):
        hidden_states = inputs[0]
        cls_index = inputs[1] if len(inputs) > 1 else None
        assert len(inputs) <= 2, 'Too many inputs.'
    else:
        input_ids = inputs.get('input_ids')
        cls_index = inputs.get('cls_index', None)
    if self.summary_type == 'last':
        output = hidden_states[:, -1]
    elif self.summary_type == 'first':
        output = hidden_states[:, 0]
    elif self.summary_type == 'mean':
        output = tf.mean(hidden_states, axis=1)
    elif self.summary_type == 'cls_index':
        hidden_shape = shape_list(hidden_states)
        if cls_index is None:
            cls_index = tf.fill(hidden_shape[:-2], hidden_shape[-2] - 1)
        cls_shape = shape_list(cls_index)
        if len(cls_shape) <= len(hidden_shape) - 2:
            cls_index = cls_index[..., tf.newaxis]
        output = tf.gather(hidden_states, cls_index, batch_dims=len(
            hidden_shape) - 2)
        output = tf.squeeze(output, axis=len(hidden_shape) - 2)
    elif self.summary_type == 'attn':
        raise NotImplementedError
    if self.has_first_dropout:
        output = self.first_dropout(output, training=training)
    if self.has_summary:
        output = self.summary(output)
    if self.has_activation:
        output = self.activation(output)
    if self.has_last_dropout:
        output = self.last_dropout(output, training=training)
    return output

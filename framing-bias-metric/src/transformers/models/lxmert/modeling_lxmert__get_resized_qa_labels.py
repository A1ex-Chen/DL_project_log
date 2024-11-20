def _get_resized_qa_labels(self, cur_qa_logit_layer, num_labels):
    if num_labels is None:
        return cur_qa_logit_layer
    cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()
    if cur_qa_labels == num_labels:
        return cur_qa_logit_layer
    if getattr(cur_qa_logit_layer, 'bias', None) is not None:
        new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
    else:
        new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)
    new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)
    self._init_weights(new_qa_logit_layer)
    num_labels_to_copy = min(cur_qa_labels, num_labels)
    new_qa_logit_layer.weight.data[:num_labels_to_copy, :
        ] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
    if getattr(cur_qa_logit_layer, 'bias', None) is not None:
        new_qa_logit_layer.bias.data[:num_labels_to_copy
            ] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]
    return new_qa_logit_layer

def set_model_weights_in_torch(weights, torch_model, hidden_size):
    torch_model_reformer = torch_model.reformer
    word_embeddings = np.asarray(weights[1])
    set_param(torch_model_reformer.embeddings.word_embeddings, torch.tensor
        (word_embeddings))
    if isinstance(weights[3], tuple):
        position_embeddings = (torch_model_reformer.embeddings.
            position_embeddings)
        for emb_idx in range(len(position_embeddings.weights)):
            emb_weights = np.asarray(weights[3][emb_idx][0])
            assert position_embeddings.weights[emb_idx
                ].shape == emb_weights.shape, '{} emb does not match'.format(
                position_embeddings[emb_idx])
            position_embeddings.weights[emb_idx] = torch.nn.Parameter(torch
                .tensor(emb_weights))
    trax_layer_weights = weights[5]
    assert len(torch_model_reformer.encoder.layers) * 4 == len(
        trax_layer_weights
        ), 'HF and trax model do not have the same number of layers'
    for layer_idx, layer in enumerate(torch_model_reformer.encoder.layers):
        block_weights = trax_layer_weights[4 * layer_idx:4 * (layer_idx + 1)]
        set_block_weights_in_torch(block_weights, layer, hidden_size)
    layer_norm_out_weight = np.asarray(weights[7][0])
    layer_norm_out_bias = np.asarray(weights[7][1])
    set_param(torch_model_reformer.encoder.layer_norm, torch.tensor(
        layer_norm_out_weight), torch.tensor(layer_norm_out_bias))
    output_embed_weights = np.asarray(weights[9][0])
    output_embed_bias = np.asarray(weights[9][1])
    set_param(torch_model.lm_head.decoder, torch.tensor(
        output_embed_weights).transpose(0, 1).contiguous(), torch.tensor(
        output_embed_bias))

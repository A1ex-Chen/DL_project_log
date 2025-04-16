def load_decoder(weights, model):
    model.conditioning_emb[0].weight = nn.Parameter(torch.Tensor(weights[
        'time_emb_dense0']['kernel'].T))
    model.conditioning_emb[2].weight = nn.Parameter(torch.Tensor(weights[
        'time_emb_dense1']['kernel'].T))
    model.position_encoding.weight = nn.Parameter(torch.Tensor(weights[
        'Embed_0']['embedding']), requires_grad=False)
    model.continuous_inputs_projection.weight = nn.Parameter(torch.Tensor(
        weights['continuous_inputs_projection']['kernel'].T))
    for lyr_num, lyr in enumerate(model.decoders):
        ly_weight = weights[f'layers_{lyr_num}']
        lyr.layer[0].layer_norm.weight = nn.Parameter(torch.Tensor(
            ly_weight['pre_self_attention_layer_norm']['scale']))
        lyr.layer[0].FiLMLayer.scale_bias.weight = nn.Parameter(torch.
            Tensor(ly_weight['FiLMLayer_0']['DenseGeneral_0']['kernel'].T))
        attention_weights = ly_weight['self_attention']
        lyr.layer[0].attention.to_q.weight = nn.Parameter(torch.Tensor(
            attention_weights['query']['kernel'].T))
        lyr.layer[0].attention.to_k.weight = nn.Parameter(torch.Tensor(
            attention_weights['key']['kernel'].T))
        lyr.layer[0].attention.to_v.weight = nn.Parameter(torch.Tensor(
            attention_weights['value']['kernel'].T))
        lyr.layer[0].attention.to_out[0].weight = nn.Parameter(torch.Tensor
            (attention_weights['out']['kernel'].T))
        attention_weights = ly_weight['MultiHeadDotProductAttention_0']
        lyr.layer[1].attention.to_q.weight = nn.Parameter(torch.Tensor(
            attention_weights['query']['kernel'].T))
        lyr.layer[1].attention.to_k.weight = nn.Parameter(torch.Tensor(
            attention_weights['key']['kernel'].T))
        lyr.layer[1].attention.to_v.weight = nn.Parameter(torch.Tensor(
            attention_weights['value']['kernel'].T))
        lyr.layer[1].attention.to_out[0].weight = nn.Parameter(torch.Tensor
            (attention_weights['out']['kernel'].T))
        lyr.layer[1].layer_norm.weight = nn.Parameter(torch.Tensor(
            ly_weight['pre_cross_attention_layer_norm']['scale']))
        lyr.layer[2].layer_norm.weight = nn.Parameter(torch.Tensor(
            ly_weight['pre_mlp_layer_norm']['scale']))
        lyr.layer[2].film.scale_bias.weight = nn.Parameter(torch.Tensor(
            ly_weight['FiLMLayer_1']['DenseGeneral_0']['kernel'].T))
        lyr.layer[2].DenseReluDense.wi_0.weight = nn.Parameter(torch.Tensor
            (ly_weight['mlp']['wi_0']['kernel'].T))
        lyr.layer[2].DenseReluDense.wi_1.weight = nn.Parameter(torch.Tensor
            (ly_weight['mlp']['wi_1']['kernel'].T))
        lyr.layer[2].DenseReluDense.wo.weight = nn.Parameter(torch.Tensor(
            ly_weight['mlp']['wo']['kernel'].T))
    model.decoder_norm.weight = nn.Parameter(torch.Tensor(weights[
        'decoder_norm']['scale']))
    model.spec_out.weight = nn.Parameter(torch.Tensor(weights[
        'spec_out_dense']['kernel'].T))
    return model

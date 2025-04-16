def load_continuous_encoder(weights, model):
    model.input_proj.weight = nn.Parameter(torch.Tensor(weights[
        'input_proj']['kernel'].T))
    model.position_encoding.weight = nn.Parameter(torch.Tensor(weights[
        'Embed_0']['embedding']), requires_grad=False)
    for lyr_num, lyr in enumerate(model.encoders):
        ly_weight = weights[f'layers_{lyr_num}']
        attention_weights = ly_weight['attention']
        lyr.layer[0].SelfAttention.q.weight = nn.Parameter(torch.Tensor(
            attention_weights['query']['kernel'].T))
        lyr.layer[0].SelfAttention.k.weight = nn.Parameter(torch.Tensor(
            attention_weights['key']['kernel'].T))
        lyr.layer[0].SelfAttention.v.weight = nn.Parameter(torch.Tensor(
            attention_weights['value']['kernel'].T))
        lyr.layer[0].SelfAttention.o.weight = nn.Parameter(torch.Tensor(
            attention_weights['out']['kernel'].T))
        lyr.layer[0].layer_norm.weight = nn.Parameter(torch.Tensor(
            ly_weight['pre_attention_layer_norm']['scale']))
        lyr.layer[1].DenseReluDense.wi_0.weight = nn.Parameter(torch.Tensor
            (ly_weight['mlp']['wi_0']['kernel'].T))
        lyr.layer[1].DenseReluDense.wi_1.weight = nn.Parameter(torch.Tensor
            (ly_weight['mlp']['wi_1']['kernel'].T))
        lyr.layer[1].DenseReluDense.wo.weight = nn.Parameter(torch.Tensor(
            ly_weight['mlp']['wo']['kernel'].T))
        lyr.layer[1].layer_norm.weight = nn.Parameter(torch.Tensor(
            ly_weight['pre_mlp_layer_norm']['scale']))
    model.layer_norm.weight = nn.Parameter(torch.Tensor(weights[
        'encoder_norm']['scale']))
    return model

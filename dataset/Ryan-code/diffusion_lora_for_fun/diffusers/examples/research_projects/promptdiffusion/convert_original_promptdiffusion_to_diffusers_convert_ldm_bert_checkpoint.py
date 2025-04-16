def convert_ldm_bert_checkpoint(checkpoint, config):

    def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
        hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
        hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
        hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight
        hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
        hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias

    def _copy_linear(hf_linear, pt_linear):
        hf_linear.weight = pt_linear.weight
        hf_linear.bias = pt_linear.bias

    def _copy_layer(hf_layer, pt_layer):
        _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
        _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])
        _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])
        pt_mlp = pt_layer[1][1]
        _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
        _copy_linear(hf_layer.fc2, pt_mlp.net[2])

    def _copy_layers(hf_layers, pt_layers):
        for i, hf_layer in enumerate(hf_layers):
            if i != 0:
                i += i
            pt_layer = pt_layers[i:i + 2]
            _copy_layer(hf_layer, pt_layer)
    hf_model = LDMBertModel(config).eval()
    hf_model.model.embed_tokens.weight = (checkpoint.transformer.token_emb.
        weight)
    hf_model.model.embed_positions.weight.data = (checkpoint.transformer.
        pos_emb.emb.weight)
    _copy_linear(hf_model.model.layer_norm, checkpoint.transformer.norm)
    _copy_layers(hf_model.model.layers, checkpoint.transformer.attn_layers.
        layers)
    _copy_linear(hf_model.to_logits, checkpoint.transformer.to_logits)
    return hf_model

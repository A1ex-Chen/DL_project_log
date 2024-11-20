def build_tf_to_pytorch_map(model, config):
    """
    A map of modules from TF to PyTorch. This time I use a map to keep the PyTorch model as identical to the original
    PyTorch model as possible.
    """
    tf_to_pt_map = {}
    if hasattr(model, 'transformer'):
        tf_to_pt_map.update({
            'transformer/adaptive_softmax/cutoff_0/cluster_W': model.crit.
            cluster_weight,
            'transformer/adaptive_softmax/cutoff_0/cluster_b': model.crit.
            cluster_bias})
        for i, (out_l, proj_l, tie_proj) in enumerate(zip(model.crit.
            out_layers, model.crit.out_projs, config.tie_projs)):
            layer_str = 'transformer/adaptive_softmax/cutoff_%d/' % i
            if config.tie_word_embeddings:
                tf_to_pt_map.update({(layer_str + 'b'): out_l.bias})
            else:
                raise NotImplementedError
                tf_to_pt_map.update({(layer_str + 'lookup_table'): out_l.
                    weight, (layer_str + 'b'): out_l.bias})
            if not tie_proj:
                tf_to_pt_map.update({(layer_str + 'proj'): proj_l})
        model = model.transformer
    for i, (embed_l, proj_l) in enumerate(zip(model.word_emb.emb_layers,
        model.word_emb.emb_projs)):
        layer_str = 'transformer/adaptive_embed/cutoff_%d/' % i
        tf_to_pt_map.update({(layer_str + 'lookup_table'): embed_l.weight,
            (layer_str + 'proj_W'): proj_l})
    for i, b in enumerate(model.layers):
        layer_str = 'transformer/layer_%d/' % i
        tf_to_pt_map.update({(layer_str + 'rel_attn/LayerNorm/gamma'): b.
            dec_attn.layer_norm.weight, (layer_str +
            'rel_attn/LayerNorm/beta'): b.dec_attn.layer_norm.bias, (
            layer_str + 'rel_attn/o/kernel'): b.dec_attn.o_net.weight, (
            layer_str + 'rel_attn/qkv/kernel'): b.dec_attn.qkv_net.weight,
            (layer_str + 'rel_attn/r/kernel'): b.dec_attn.r_net.weight, (
            layer_str + 'ff/LayerNorm/gamma'): b.pos_ff.layer_norm.weight,
            (layer_str + 'ff/LayerNorm/beta'): b.pos_ff.layer_norm.bias, (
            layer_str + 'ff/layer_1/kernel'): b.pos_ff.CoreNet[0].weight, (
            layer_str + 'ff/layer_1/bias'): b.pos_ff.CoreNet[0].bias, (
            layer_str + 'ff/layer_2/kernel'): b.pos_ff.CoreNet[3].weight, (
            layer_str + 'ff/layer_2/bias'): b.pos_ff.CoreNet[3].bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        for b in model.layers:
            r_r_list.append(b.dec_attn.r_r_bias)
            r_w_list.append(b.dec_attn.r_w_bias)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
    tf_to_pt_map.update({'transformer/r_r_bias': r_r_list,
        'transformer/r_w_bias': r_w_list})
    return tf_to_pt_map

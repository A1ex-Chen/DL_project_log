def build_tf_xlnet_to_pytorch_map(model, config, tf_weights=None):
    """ A map of modules from TF to PyTorch.
        I use a map to keep the PyTorch model as
        identical to the original PyTorch model as possible.
    """
    tf_to_pt_map = {}
    if hasattr(model, 'transformer'):
        if hasattr(model, 'lm_loss'):
            tf_to_pt_map['model/lm_loss/bias'] = model.lm_loss.bias
        if hasattr(model, 'sequence_summary'
            ) and 'model/sequnece_summary/summary/kernel' in tf_weights:
            tf_to_pt_map['model/sequnece_summary/summary/kernel'
                ] = model.sequence_summary.summary.weight
            tf_to_pt_map['model/sequnece_summary/summary/bias'
                ] = model.sequence_summary.summary.bias
        if (hasattr(model, 'logits_proj') and config.finetuning_task is not
            None and 'model/regression_{}/logit/kernel'.format(config.
            finetuning_task) in tf_weights):
            tf_to_pt_map['model/regression_{}/logit/kernel'.format(config.
                finetuning_task)] = model.logits_proj.weight
            tf_to_pt_map['model/regression_{}/logit/bias'.format(config.
                finetuning_task)] = model.logits_proj.bias
        model = model.transformer
    tf_to_pt_map.update({'model/transformer/word_embedding/lookup_table':
        model.word_embedding.weight, 'model/transformer/mask_emb/mask_emb':
        model.mask_emb})
    for i, b in enumerate(model.layer):
        layer_str = 'model/transformer/layer_%d/' % i
        tf_to_pt_map.update({(layer_str + 'rel_attn/LayerNorm/gamma'): b.
            rel_attn.layer_norm.weight, (layer_str +
            'rel_attn/LayerNorm/beta'): b.rel_attn.layer_norm.bias, (
            layer_str + 'rel_attn/o/kernel'): b.rel_attn.o, (layer_str +
            'rel_attn/q/kernel'): b.rel_attn.q, (layer_str +
            'rel_attn/k/kernel'): b.rel_attn.k, (layer_str +
            'rel_attn/r/kernel'): b.rel_attn.r, (layer_str +
            'rel_attn/v/kernel'): b.rel_attn.v, (layer_str +
            'ff/LayerNorm/gamma'): b.ff.layer_norm.weight, (layer_str +
            'ff/LayerNorm/beta'): b.ff.layer_norm.bias, (layer_str +
            'ff/layer_1/kernel'): b.ff.layer_1.weight, (layer_str +
            'ff/layer_1/bias'): b.ff.layer_1.bias, (layer_str +
            'ff/layer_2/kernel'): b.ff.layer_2.weight, (layer_str +
            'ff/layer_2/bias'): b.ff.layer_2.bias})
    if config.untie_r:
        r_r_list = []
        r_w_list = []
        r_s_list = []
        seg_embed_list = []
        for b in model.layer:
            r_r_list.append(b.rel_attn.r_r_bias)
            r_w_list.append(b.rel_attn.r_w_bias)
            r_s_list.append(b.rel_attn.r_s_bias)
            seg_embed_list.append(b.rel_attn.seg_embed)
    else:
        r_r_list = [model.r_r_bias]
        r_w_list = [model.r_w_bias]
        r_s_list = [model.r_s_bias]
        seg_embed_list = [model.seg_embed]
    tf_to_pt_map.update({'model/transformer/r_r_bias': r_r_list,
        'model/transformer/r_w_bias': r_w_list,
        'model/transformer/r_s_bias': r_s_list,
        'model/transformer/seg_embed': seg_embed_list})
    return tf_to_pt_map

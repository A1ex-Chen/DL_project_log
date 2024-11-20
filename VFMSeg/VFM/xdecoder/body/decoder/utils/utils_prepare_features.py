def prepare_features(x, num_feature_levels, pe_layer, input_proj, level_embed):
    src = []
    pos = []
    size_list = []
    for i in range(num_feature_levels):
        size_list.append(x[i].shape[-2:])
        pos.append(pe_layer(x[i], None).flatten(2))
        src.append(input_proj[i](x[i]).flatten(2) + level_embed.weight[i][
            None, :, None])
        pos[-1] = pos[-1].permute(2, 0, 1)
        src[-1] = src[-1].permute(2, 0, 1)
    return src, pos, size_list

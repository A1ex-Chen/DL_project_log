def forward(self, x, mask_features, mask=None, target_queries=None,
    target_vlp=None, task='seg', extra={}):
    assert len(x) == self.num_feature_levels
    del mask
    spatial_extra_flag = 'spatial_query_pos_mask' in extra.keys(
        ) or task == 'refimg'
    grounding_extra_flag = 'grounding_tokens' in extra.keys()
    visual_extra_flag = 'visual_query_pos' in extra.keys()
    audio_extra_flag = 'audio_tokens' in extra.keys()
    spatial_memory_flag = 'prev_mask' in extra.keys()
    flags = {'spatial': spatial_extra_flag, 'grounding':
        grounding_extra_flag, 'memories_spatial': spatial_memory_flag,
        'visual': visual_extra_flag, 'audio': audio_extra_flag}
    self.attention_data.reset(flags, task, extra)
    src, pos, size_list = prepare_features(x, self.num_feature_levels, self
        .pe_layer, self.input_proj, self.level_embed)
    _, bs, _ = src[0].shape
    query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
    output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
    self.attention_data.set('queries_object', 'queries', output, query_embed)
    if self.task_switch['spatial'] and spatial_extra_flag:
        _, h, w = extra['spatial_query_pos_mask'][0].shape
        divisor = torch.tensor([h, w], device=output.device)[None,]
        non_zero_pos_point = [rand_sample((m.nonzero()[:, 1:] / divisor).t(
            ), self.max_spatial_len[-1]).t() for m in extra[
            'spatial_query_pos_mask']]
        non_zero_pos_point = nn.utils.rnn.pad_sequence(non_zero_pos_point,
            padding_value=-1).permute(1, 0, 2)
        non_zero_pos_mask = non_zero_pos_point.sum(dim=-1) < 0
        spatial_query_pos = point_sample(mask_features, non_zero_pos_point.
            flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
        spatial_query_pos = torch.stack([x[m].mean(dim=0, keepdim=True) for
            x, m in zip(spatial_query_pos.transpose(1, 2), ~non_zero_pos_mask)]
            ).transpose(0, 1).nan_to_num()
        non_zero_neg_point = [rand_sample((m.nonzero()[:, 1:] / divisor).t(
            ), self.max_spatial_len[-1]).t() for m in extra[
            'spatial_query_neg_mask']]
        non_zero_neg_point = nn.utils.rnn.pad_sequence(non_zero_neg_point,
            padding_value=-1).permute(1, 0, 2)
        non_zero_neg_mask = non_zero_neg_point.sum(dim=-1) < 0
        spatial_query_neg = point_sample(mask_features, non_zero_neg_point.
            flip(dims=(2,)).type(mask_features.dtype), align_corners=True)
        spatial_query_neg = torch.stack([x[m].mean(dim=0, keepdim=True) for
            x, m in zip(spatial_query_neg.transpose(1, 2), ~non_zero_neg_mask)]
            ).transpose(0, 1).nan_to_num()
        src_spatial_queries = []
        src_spatial_maskings = []
        for i in range(len(src)):
            hw, _, dc = src[i].shape
            src_mask_features = src[i].view(size_list[i][0], size_list[i][1
                ], bs, dc)
            src_mask_features = src_mask_features @ self.mask_sptial_embed[i]
            non_zero_query_point_pos = [rand_sample((m.nonzero()[:, 1:] /
                divisor).t(), self.max_spatial_len[i]).t() for m in extra[
                'spatial_query_pos_mask']]
            non_zero_query_point_neg = [rand_sample((m.nonzero()[:, 1:] /
                divisor).t(), self.max_spatial_len[i]).t() for m in extra[
                'spatial_query_neg_mask']]
            non_zero_query_point = [torch.cat([x, y], dim=0) for x, y in
                zip(non_zero_query_point_pos, non_zero_query_point_neg)]
            pos_neg_indicator = [torch.cat([torch.ones(x.shape[0], device=x
                .device), -torch.ones(y.shape[0], device=y.device)]) for x,
                y in zip(non_zero_query_point_pos, non_zero_query_point_neg)]
            pos_neg_indicator = nn.utils.rnn.pad_sequence(pos_neg_indicator,
                padding_value=0)
            non_zero_query_point = nn.utils.rnn.pad_sequence(
                non_zero_query_point, padding_value=-1).permute(1, 0, 2)
            non_zero_query_mask = non_zero_query_point.sum(dim=-1) < 0
            non_zero_query_point[non_zero_query_mask] = 0
            spatial_tokens = point_sample(src_mask_features.permute(2, 3, 0,
                1), non_zero_query_point.flip(dims=(2,)).type(
                src_mask_features.dtype), align_corners=True).permute(2, 0, 1)
            spatial_tokens[pos_neg_indicator == 1] += self.pn_indicator.weight[
                0:1]
            spatial_tokens[pos_neg_indicator == -1
                ] += self.pn_indicator.weight[1:2]
            src_spatial_queries += [spatial_tokens]
            src_spatial_maskings += [non_zero_query_mask]
        if 'refimg' in task:
            output_refimg = {}
            output_refimg['visual_query_pos'] = spatial_query_pos
            output_refimg['visual_query_neg'] = spatial_query_neg
            output_refimg['src_visual_queries'] = src_spatial_queries
            output_refimg['src_visual_maskings'] = src_spatial_maskings
            return output_refimg
        if task != 'demo':
            self.attention_data.set('queries_spatial', 'queries')
    if self.task_switch['visual'] and visual_extra_flag:
        visual_query_pos = extra['visual_query_pos']
        visual_query_neg = extra['visual_query_neg']
        src_visual_queries = extra['src_visual_queries']
        src_visual_maskings = extra['src_visual_maskings']
    if self.task_switch['grounding'] and grounding_extra_flag:
        grounding_tokens = extra['grounding_tokens']
        _grounding_tokens = grounding_tokens.detach().clone()
        self.attention_data.set('tokens_grounding', 'tokens',
            grounding_tokens, _grounding_tokens)
        self.attention_data.set_maskings('tokens_grounding', extra[
            'grounding_nonzero_mask'])
    if self.task_switch['audio'] and audio_extra_flag:
        grounding_tokens = extra['audio_tokens']
        _grounding_tokens = grounding_tokens.detach().clone()
        self.attention_data.set('tokens_audio', 'tokens', grounding_tokens,
            _grounding_tokens)
        self.attention_data.set_maskings('tokens_audio', extra[
            'audio_nonzero_mask'])
    output, query_embed = self.attention_data.cross_attn_variables()
    results = self.forward_prediction_heads(output, mask_features,
        attn_mask_target_size=size_list[0])
    results['predictions_pos_spatial'] = spatial_query_pos.transpose(0, 1
        ) if spatial_extra_flag else None
    results['predictions_neg_spatial'] = spatial_query_neg.transpose(0, 1
        ) if spatial_extra_flag else None
    results['predictions_pos_visual'] = visual_query_pos.transpose(0, 1
        ) if visual_extra_flag else None
    results['predictions_neg_visual'] = visual_query_neg.transpose(0, 1
        ) if visual_extra_flag else None
    self.attention_data.set_results(results)
    for i in range(self.num_layers):
        level_index = i % self.num_feature_levels
        output, avg_attn = self.transformer_cross_attention_layers[i](output,
            src[level_index], memory_mask=self.attention_data.
            cross_attn_mask(size_list[level_index], self.num_heads),
            memory_key_padding_mask=None, pos=pos[level_index], query_pos=
            query_embed)
        self.attention_data.update_variables(output, 'cross_attn')
        self_attn_mask = torch.zeros((bs, self.num_queries, self.
            num_queries), device=query_embed.device).bool()
        if self.task_switch['spatial'] and spatial_extra_flag:
            spatial_tokens = src_spatial_queries[level_index]
            _spatial_tokens = spatial_tokens.detach().clone()
            self.attention_data.set('tokens_spatial', 'tokens',
                spatial_tokens, _spatial_tokens)
            self.attention_data.set_maskings('tokens_spatial',
                src_spatial_maskings[level_index])
        if self.task_switch['visual'] and visual_extra_flag:
            visual_tokens = src_visual_queries[level_index]
            _visual_tokens = visual_tokens.detach().clone()
            self.attention_data.set('tokens_visual', 'tokens',
                visual_tokens, _visual_tokens)
            self.attention_data.set_maskings('tokens_visual',
                src_visual_maskings[level_index])
        output, query_embed, self_attn_mask = self.attention_data.self_attn(bs,
            self.num_heads)
        output = self.transformer_self_attention_layers[i](output, tgt_mask
            =self_attn_mask, tgt_key_padding_mask=None, query_pos=query_embed)
        output = self.transformer_ffn_layers[i](output)
        self.attention_data.update_variables(output, 'self_attn')
        output, query_embed = self.attention_data.cross_attn_variables()
        results = self.forward_prediction_heads(output, mask_features,
            attn_mask_target_size=size_list[(i + 1) % self.
            num_feature_levels], layer_id=i)
        results['predictions_pos_spatial'] = spatial_query_pos.transpose(0, 1
            ) if spatial_extra_flag else None
        results['predictions_neg_spatial'] = spatial_query_neg.transpose(0, 1
            ) if spatial_extra_flag else None
        results['predictions_pos_visual'] = visual_query_pos.transpose(0, 1
            ) if visual_extra_flag else None
        results['predictions_neg_visual'] = visual_query_neg.transpose(0, 1
            ) if visual_extra_flag else None
        self.attention_data.set_results(results)
    return self.attention_data.organize_output()

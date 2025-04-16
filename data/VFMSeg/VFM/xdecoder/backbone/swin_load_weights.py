def load_weights(self, pretrained_dict=None, pretrained_layers=[], verbose=True
    ):
    model_dict = self.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
        model_dict.keys()}
    need_init_state_dict = {}
    for k, v in pretrained_dict.items():
        need_init = (k.split('.')[0] in pretrained_layers or 
            pretrained_layers[0] == '*'
            ) and 'relative_position_index' not in k and 'attn_mask' not in k
        if need_init:
            if 'relative_position_bias_table' in k and v.size() != model_dict[k
                ].size():
                relative_position_bias_table_pretrained = v
                relative_position_bias_table_current = model_dict[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.info(f'Error in loading {k}, passing')
                elif L1 != L2:
                    logger.info('=> load_pretrained: resized variant: {} to {}'
                        .format((L1, nH1), (L2, nH2)))
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    relative_position_bias_table_pretrained_resized = (torch
                        .nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 
                        0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                        )
                    v = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)
            if 'absolute_pos_embed' in k and v.size() != model_dict[k].size():
                absolute_pos_embed_pretrained = v
                absolute_pos_embed_current = model_dict[k]
                _, L1, C1 = absolute_pos_embed_pretrained.size()
                _, L2, C2 = absolute_pos_embed_current.size()
                if C1 != C1:
                    logger.info(f'Error in loading {k}, passing')
                elif L1 != L2:
                    logger.info('=> load_pretrained: resized variant: {} to {}'
                        .format((1, L1, C1), (1, L2, C2)))
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    absolute_pos_embed_pretrained = (
                        absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1))
                    absolute_pos_embed_pretrained = (
                        absolute_pos_embed_pretrained.permute(0, 3, 1, 2))
                    absolute_pos_embed_pretrained_resized = (torch.nn.
                        functional.interpolate(
                        absolute_pos_embed_pretrained, size=(S2, S2), mode=
                        'bicubic'))
                    v = absolute_pos_embed_pretrained_resized.permute(0, 2,
                        3, 1).flatten(1, 2)
            need_init_state_dict[k] = v
    self.load_state_dict(need_init_state_dict, strict=False)

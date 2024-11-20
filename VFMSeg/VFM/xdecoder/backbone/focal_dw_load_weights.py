def load_weights(self, pretrained_dict=None, pretrained_layers=[], verbose=True
    ):
    model_dict = self.state_dict()
    missed_dict = [k for k in model_dict.keys() if k not in pretrained_dict]
    logger.info(f'=> Missed keys {missed_dict}')
    unexpected_dict = [k for k in pretrained_dict.keys() if k not in model_dict
        ]
    logger.info(f'=> Unexpected keys {unexpected_dict}')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
        model_dict.keys()}
    need_init_state_dict = {}
    for k, v in pretrained_dict.items():
        need_init = (k.split('.')[0] in pretrained_layers or 
            pretrained_layers[0] == '*'
            ) and 'relative_position_index' not in k and 'attn_mask' not in k
        if need_init:
            if 'pool_layers' in k or 'focal_layers' in k and v.size(
                ) != model_dict[k].size():
                table_pretrained = v
                table_current = model_dict[k]
                fsize1 = table_pretrained.shape[2]
                fsize2 = table_current.shape[2]
                if fsize1 < fsize2:
                    table_pretrained_resized = torch.zeros(table_current.shape)
                    table_pretrained_resized[:, :, (fsize2 - fsize1) // 2:-
                        (fsize2 - fsize1) // 2, (fsize2 - fsize1) // 2:-(
                        fsize2 - fsize1) // 2] = table_pretrained
                    v = table_pretrained_resized
                elif fsize1 > fsize2:
                    table_pretrained_resized = table_pretrained[:, :, (
                        fsize1 - fsize2) // 2:-(fsize1 - fsize2) // 2, (
                        fsize1 - fsize2) // 2:-(fsize1 - fsize2) // 2]
                    v = table_pretrained_resized
            if 'modulation.f' in k or 'pre_conv' in k:
                table_pretrained = v
                table_current = model_dict[k]
                if table_pretrained.shape != table_current.shape:
                    if len(table_pretrained.shape) == 2:
                        dim = table_pretrained.shape[1]
                        assert table_current.shape[1] == dim
                        L1 = table_pretrained.shape[0]
                        L2 = table_current.shape[0]
                        if L1 < L2:
                            table_pretrained_resized = torch.zeros(
                                table_current.shape)
                            table_pretrained_resized[:2 * dim
                                ] = table_pretrained[:2 * dim]
                            table_pretrained_resized[-1] = table_pretrained[-1]
                            table_pretrained_resized[2 * dim:2 * dim + (L1 -
                                2 * dim - 1)] = table_pretrained[2 * dim:-1]
                            v = table_pretrained_resized
                        elif L1 > L2:
                            raise NotImplementedError
                    elif len(table_pretrained.shape) == 1:
                        dim = table_pretrained.shape[0]
                        L1 = table_pretrained.shape[0]
                        L2 = table_current.shape[0]
                        if L1 < L2:
                            table_pretrained_resized = torch.zeros(
                                table_current.shape)
                            table_pretrained_resized[:dim] = table_pretrained[:
                                dim]
                            table_pretrained_resized[-1] = table_pretrained[-1]
                            v = table_pretrained_resized
                        elif L1 > L2:
                            raise NotImplementedError
            need_init_state_dict[k] = v
    self.load_state_dict(need_init_state_dict, strict=False)

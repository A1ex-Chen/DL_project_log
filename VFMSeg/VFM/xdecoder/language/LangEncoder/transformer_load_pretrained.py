def load_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
    if os.path.isfile(pretrained):
        pretrained_dict = torch.load(pretrained, map_location='cpu')
        logging.info(f'=> loading pretrained model {pretrained}')
        model_dict = self.state_dict()
        stripped_key = lambda x: x[13:] if x.startswith('lang_encoder.') else x
        pretrained_dict = {stripped_key(k): v for k, v in pretrained_dict.
            items() if stripped_key(k) in model_dict.keys()}
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = k.split('.')[0
                ] in pretrained_layers or pretrained_layers[0] == '*'
            if need_init:
                if verbose:
                    logger.info(f'=> init {k} from {pretrained}')
                if 'positional_embedding' in k and v.size() != model_dict[k
                    ].size():
                    positional_embedding_pretrained = v
                    positional_embedding_current = model_dict[k]
                    L1, nH1 = positional_embedding_pretrained.size()
                    L2, nH2 = positional_embedding_current.size()
                    if nH1 != nH2:
                        logger.info(f'Error in loading {k}, passing')
                    elif L1 != L2:
                        logger.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format((L1, nH1), (L2, nH2)))
                        posemb = positional_embedding_pretrained.float()
                        posemb_grid = posemb.unsqueeze(dim=0).permute(0, 2, 1)
                        posemb_grid = torch.nn.functional.interpolate(
                            posemb_grid, size=L2, mode='linear')
                        posemb_grid = posemb_grid.permute(0, 2, 1).squeeze(dim
                            =0)
                        v = posemb_grid
                need_init_state_dict[k] = v
        self.load_state_dict(need_init_state_dict, strict=False)

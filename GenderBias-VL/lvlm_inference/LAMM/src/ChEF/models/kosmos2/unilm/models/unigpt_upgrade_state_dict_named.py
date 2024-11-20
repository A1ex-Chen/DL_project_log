def upgrade_state_dict_named(self, state_dict, name):
    prefix = name + '.' if name != '' else ''
    super().upgrade_state_dict_named(state_dict, name)
    current_head_names = [] if not hasattr(self, 'classification_heads'
        ) else self.classification_heads.keys()
    keys_to_delete = []
    for k in state_dict.keys():
        if not k.startswith(prefix + 'classification_heads.'):
            continue
        head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
        num_classes = state_dict[prefix + 'classification_heads.' +
            head_name + '.out_proj.weight'].size(0)
        inner_dim = state_dict[prefix + 'classification_heads.' + head_name +
            '.dense.weight'].size(0)
        if getattr(self.args, 'load_checkpoint_heads', False):
            if head_name not in current_head_names:
                self.register_classification_head(head_name, num_classes,
                    inner_dim)
        elif head_name not in current_head_names:
            logger.warning(
                'deleting classification head ({}) from checkpoint not present in current model: {}'
                .format(head_name, k))
            keys_to_delete.append(k)
        elif num_classes != self.classification_heads[head_name
            ].out_proj.out_features or inner_dim != self.classification_heads[
            head_name].dense.out_features:
            logger.warning(
                'deleting classification head ({}) from checkpoint with different dimensions than current model: {}'
                .format(head_name, k))
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del state_dict[k]
    if hasattr(self, 'classification_heads'):
        cur_state = self.classification_heads.state_dict()
        for k, v in cur_state.items():
            if prefix + 'classification_heads.' + k not in state_dict:
                logger.info('Overwriting ' + prefix +
                    'classification_heads.' + k)
                state_dict[prefix + 'classification_heads.' + k] = v

def get_optimizer_params(self, weight_decay, lr_scale=1):
    if self.vit_name == 'eva_clip_g':
        vit_num_layers = self.visual_encoder.get_num_layer()
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in
            range(vit_num_layers + 2))
        parameter_group_names = {}
        parameter_group_vars = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith('.bias'):
                group_name = 'no_decay'
                this_weight_decay = 0.0
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace(
                    'visual_encoder.', ''))
                group_name = 'vit_layer_%d_%s' % (layer_id, group_name)
            else:
                layer_id = None
            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {'weight_decay':
                    this_weight_decay, 'params': [], 'lr_scale': scale}
                parameter_group_vars[group_name] = {'weight_decay':
                    this_weight_decay, 'params': [], 'lr_scale': scale}
            parameter_group_vars[group_name]['params'].append(param)
            parameter_group_names[group_name]['params'].append(name)
        optim_params = list(parameter_group_vars.values())
        return optim_params
    else:
        return super().get_optimizer_params(weight_decay, lr_scale)

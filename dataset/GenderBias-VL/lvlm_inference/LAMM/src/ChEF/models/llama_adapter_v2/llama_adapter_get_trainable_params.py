def get_trainable_params(self, phase='finetune'):
    for name, para in self.named_parameters():
        para.requires_grad = False
    if phase == 'finetune':
        for name, para in self.named_parameters():
            if name.startswith('llama.'):
                if 'norm' in name or 'bias' in name:
                    para.data = para.data.float()
                    para.requires_grad = True
    elif phase == 'pretrain':
        train_param_name = ['gate', 'clip_proj', 'clip_proj_norm',
            'visual_query', 'visual_blocks', 'visual_proj',
            'visual_proj_norm', 'adapter_query']
        for name, para in self.named_parameters():
            for train_name in train_param_name:
                if train_name in name:
                    para.data = para.data.float()
                    para.requires_grad = True
    else:
        raise ValueError(f'Unknown model phase: {phase}')

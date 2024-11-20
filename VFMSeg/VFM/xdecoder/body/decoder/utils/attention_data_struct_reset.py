def reset(self, flags, task, extra):
    self.attn_variables = {}
    self.cross_attn_dict = {}
    self.self_attn_dict = {}
    self.duplication_dict = {}
    self.query_index = {}
    self.output = {}
    self.flags = {}
    self.spatial_memory = {}
    for key, values in self.p_duplication.items():
        for name in values:
            self.duplication_dict['{}_{}'.format(key, name)
                ] = self.p_duplication[key][name]
    self.flags = {'object': True}
    self.flags.update(flags)
    self.task = task
    if self.task_switch['mask']:
        self.output['predictions_class'] = []
        self.output['predictions_mask'] = []
        self.output['predictions_maskemb'] = []
    if self.task_switch['bbox']:
        self.output['predictions_bbox'] = []
    if self.task_switch['spatial'] and ('spatial' in self.flags and self.
        flags['spatial'] == True):
        self.output['predictions_pos_spatial'] = []
        self.output['predictions_neg_spatial'] = []
    if self.task_switch['spatial'] and ('memories_spatial' in self.flags and
        self.flags['memories_spatial'] == True):
        self.spatial_memory['prev_batch_mask'] = extra['prev_mask']
    if self.task_switch['grounding'] and ('grounding' in self.flags and 
        self.flags['grounding'] == True) or self.task_switch['audio'] and (
        'audio' in self.flags and self.flags['audio'] == True):
        self.output['predictions_caption'] = []
    if self.task_switch['visual']:
        self.output['predictions_pos_visual'] = []
        self.output['predictions_neg_visual'] = []
    for key, values in self.p_cross_attn.items():
        for name in values:
            self.cross_attn_dict['{}_{}'.format(key, name)
                ] = self.p_cross_attn[key][name]
    for key, values in self.p_self_attn.items():
        for name in values:
            self.self_attn_dict['{}_{}'.format(key, name)] = self.p_self_attn[
                key][name]
    self.masking = self.p_masking
    self.query_index = {'all': [0, None]}

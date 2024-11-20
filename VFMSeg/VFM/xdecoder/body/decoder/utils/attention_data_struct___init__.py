def __init__(self, attn_arch, task_switch):
    super(AttentionDataStruct, self).__init__()
    self.task_switch = task_switch
    self.p_attn_variables = attn_arch['VARIABLE']
    self.p_self_attn = attn_arch['SELF_ATTENTION']
    self.p_cross_attn = attn_arch['CROSS_ATTENTION']
    self.p_masking = attn_arch['MASKING']
    self.p_duplication = attn_arch['DUPLICATION']
    self.num_layers = attn_arch['NUM_LAYERS']

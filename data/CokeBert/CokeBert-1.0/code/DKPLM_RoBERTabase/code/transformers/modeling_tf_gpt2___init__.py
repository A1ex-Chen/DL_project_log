def __init__(self, config, *inputs, **kwargs):
    super(TFGPT2DoubleHeadsModel, self).__init__(config, *inputs, **kwargs)
    self.transformer = TFGPT2MainLayer(config, name='transformer')
    self.multiple_choice_head = TFSequenceSummary(config, initializer_range
        =config.initializer_range, name='multiple_choice_head')

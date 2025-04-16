def __init__(self, config, *inputs, **kwargs):
    super().__init__(config, *inputs, **kwargs)
    config.num_labels = 1
    self.transformer = TFOpenAIGPTMainLayer(config, name='transformer')
    self.multiple_choice_head = TFSequenceSummary(config, initializer_range
        =config.initializer_range, name='multiple_choice_head')

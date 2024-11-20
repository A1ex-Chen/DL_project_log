def __init__(self, config, *inputs, **kwargs):
    super(TFOpenAIGPTDoubleHeadsModel, self).__init__(config, *inputs, **kwargs
        )
    self.transformer = TFOpenAIGPTMainLayer(config, name='transformer')
    self.multiple_choice_head = TFSequenceSummary(config, initializer_range
        =config.initializer_range, name='multiple_choice_head')

def forward(self, input_ids, **kwargs):
    return self.model(input_ids, **kwargs)

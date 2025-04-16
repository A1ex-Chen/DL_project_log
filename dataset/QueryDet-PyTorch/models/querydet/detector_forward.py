def forward(self, batched_inputs, just_forward=False):
    if self.training:
        return self.train_forward(batched_inputs, just_forward)
    else:
        return self.test(batched_inputs)

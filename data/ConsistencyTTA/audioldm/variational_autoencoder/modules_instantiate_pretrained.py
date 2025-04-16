def instantiate_pretrained(self, config):
    model = instantiate_from_config(config)
    self.pretrained_model = model.eval()
    for param in self.pretrained_model.parameters():
        param.requires_grad = False

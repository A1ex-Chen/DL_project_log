@property
def output_module_weight(self) ->Tensor:
    return self.net.classifier[1].weight.detach()

def freeze_base_params(self) ->None:
    """Freeze the weights of the parts belonging to the base UNet2DConditionModel, and leave everything else unfrozen for fine
        tuning."""
    for param in self.parameters():
        param.requires_grad = True
    base_parts = [self.resnets]
    if isinstance(self.attentions, nn.ModuleList):
        base_parts.append(self.attentions)
    if self.upsamplers is not None:
        base_parts.append(self.upsamplers)
    for part in base_parts:
        for param in part.parameters():
            param.requires_grad = False

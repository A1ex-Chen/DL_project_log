def _make_layer(self, planes, blocks, stride=1):
    layers = [Bottleneck(self._inplanes, planes, stride)]
    self._inplanes = planes * Bottleneck.expansion
    for _ in range(1, blocks):
        layers.append(Bottleneck(self._inplanes, planes))
    return nn.Sequential(*layers)

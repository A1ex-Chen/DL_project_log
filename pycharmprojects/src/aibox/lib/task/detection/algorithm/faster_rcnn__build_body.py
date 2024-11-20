def _build_body(self) ->Tuple[nn.Module, int]:
    body = nn.Sequential(self.backbone.component.conv1, self.backbone.
        component.conv2, self.backbone.component.conv3, self.backbone.
        component.conv4)
    num_body_out = self.backbone.component.num_conv4_out
    return body, num_body_out

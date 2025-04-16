def forward(self, x):
    export_mode = torch.onnx.is_in_onnx_export() or self.export
    x = self.backbone(x)
    x = self.neck(x)
    if not export_mode:
        featmaps = []
        featmaps.extend(x)
    x = self.detect(x)
    return x if export_mode or self.export is True else [x, featmaps]

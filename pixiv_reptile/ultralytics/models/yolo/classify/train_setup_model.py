def setup_model(self):
    """Load, create or download model for any task."""
    import torchvision
    if str(self.model) in torchvision.models.__dict__:
        self.model = torchvision.models.__dict__[self.model](weights=
            'IMAGENET1K_V1' if self.args.pretrained else None)
        ckpt = None
    else:
        ckpt = super().setup_model()
    ClassificationModel.reshape_outputs(self.model, self.data['nc'])
    return ckpt

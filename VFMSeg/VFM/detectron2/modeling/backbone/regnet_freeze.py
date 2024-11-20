def freeze(self, freeze_at=0):
    """
        Freeze the first several stages of the model. Commonly used in fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this model itself
        """
    if freeze_at >= 1:
        self.stem.freeze()
    for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
        if freeze_at >= idx:
            for block in stage.children():
                block.freeze()
    return self
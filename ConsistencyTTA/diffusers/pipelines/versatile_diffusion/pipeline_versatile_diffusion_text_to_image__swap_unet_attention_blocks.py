def _swap_unet_attention_blocks(self):
    """
        Swap the `Transformer2DModel` blocks between the image and text UNets
        """
    for name, module in self.image_unet.named_modules():
        if isinstance(module, Transformer2DModel):
            parent_name, index = name.rsplit('.', 1)
            index = int(index)
            self.image_unet.get_submodule(parent_name)[index
                ], self.text_unet.get_submodule(parent_name)[index
                ] = self.text_unet.get_submodule(parent_name)[index
                ], self.image_unet.get_submodule(parent_name)[index]

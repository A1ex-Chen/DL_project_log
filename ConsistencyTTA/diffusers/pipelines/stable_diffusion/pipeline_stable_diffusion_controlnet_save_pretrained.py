def save_pretrained(self, save_directory: Union[str, os.PathLike],
    safe_serialization: bool=False, variant: Optional[str]=None):
    if isinstance(self.controlnet, ControlNetModel):
        super().save_pretrained(save_directory, safe_serialization, variant)
    else:
        raise NotImplementedError(
            'Currently, the `save_pretrained()` is not implemented for Multi-ControlNet.'
            )

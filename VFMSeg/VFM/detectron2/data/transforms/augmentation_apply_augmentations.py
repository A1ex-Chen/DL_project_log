def apply_augmentations(self, augmentations: List[Union[Augmentation,
    Transform]]) ->TransformList:
    """
        Equivalent of ``AugmentationList(augmentations)(self)``
        """
    return AugmentationList(augmentations)(self)

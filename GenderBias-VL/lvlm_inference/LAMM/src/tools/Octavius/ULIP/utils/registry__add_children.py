def _add_children(self, registry):
    """Add children for a registry.
        The ``registry`` will be added as children based on its scope.
        The parent registry could build objects from children registry.
        Example:
            >>> models = Registry('models')
            >>> mmdet_models = Registry('models', parent=models)
            >>> @mmdet_models.register_module()
            >>> class ResNet:
            >>>     pass
            >>> resnet = models.build(dict(NAME='mmdet.ResNet'))
        """
    assert isinstance(registry, Registry)
    assert registry.scope is not None
    assert registry.scope not in self.children, f'scope {registry.scope} exists in {self.name} registry'
    self.children[registry.scope] = registry

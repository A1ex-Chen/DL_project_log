def save_pretrained(self, save_directory: Union[str, os.PathLike],
    safe_serialization: bool=False, variant: Optional[str]=None):
    """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
    self.save_config(save_directory)
    model_index_dict = dict(self.config)
    model_index_dict.pop('_class_name')
    model_index_dict.pop('_diffusers_version')
    model_index_dict.pop('_module', None)
    expected_modules, optional_kwargs = self._get_signature_keys(self)

    def is_saveable_module(name, value):
        if name not in expected_modules:
            return False
        if name in self._optional_components and value[0] is None:
            return False
        return True
    model_index_dict = {k: v for k, v in model_index_dict.items() if
        is_saveable_module(k, v)}
    for pipeline_component_name in model_index_dict.keys():
        sub_model = getattr(self, pipeline_component_name)
        model_cls = sub_model.__class__
        if is_compiled_module(sub_model):
            sub_model = sub_model._orig_mod
            model_cls = sub_model.__class__
        save_method_name = None
        for library_name, library_classes in LOADABLE_CLASSES.items():
            library = importlib.import_module(library_name)
            for base_class, save_load_methods in library_classes.items():
                class_candidate = getattr(library, base_class, None)
                if class_candidate is not None and issubclass(model_cls,
                    class_candidate):
                    save_method_name = save_load_methods[0]
                    break
            if save_method_name is not None:
                break
        save_method = getattr(sub_model, save_method_name)
        save_method_signature = inspect.signature(save_method)
        save_method_accept_safe = ('safe_serialization' in
            save_method_signature.parameters)
        save_method_accept_variant = ('variant' in save_method_signature.
            parameters)
        save_kwargs = {}
        if save_method_accept_safe:
            save_kwargs['safe_serialization'] = safe_serialization
        if save_method_accept_variant:
            save_kwargs['variant'] = variant
        save_method(os.path.join(save_directory, pipeline_component_name),
            **save_kwargs)

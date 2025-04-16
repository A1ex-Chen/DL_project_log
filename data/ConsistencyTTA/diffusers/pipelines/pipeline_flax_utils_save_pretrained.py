def save_pretrained(self, save_directory: Union[str, os.PathLike], params:
    Union[Dict, FrozenDict]):
    """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~FlaxDiffusionPipeline.from_pretrained`]` class
        method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
    self.save_config(save_directory)
    model_index_dict = dict(self.config)
    model_index_dict.pop('_class_name')
    model_index_dict.pop('_diffusers_version')
    model_index_dict.pop('_module', None)
    for pipeline_component_name in model_index_dict.keys():
        sub_model = getattr(self, pipeline_component_name)
        if sub_model is None:
            continue
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
        expects_params = 'params' in set(inspect.signature(save_method).
            parameters.keys())
        if expects_params:
            save_method(os.path.join(save_directory,
                pipeline_component_name), params=params[
                pipeline_component_name])
        else:
            save_method(os.path.join(save_directory, pipeline_component_name))

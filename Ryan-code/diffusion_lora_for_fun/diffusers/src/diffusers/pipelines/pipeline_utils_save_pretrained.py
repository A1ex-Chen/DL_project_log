def save_pretrained(self, save_directory: Union[str, os.PathLike],
    safe_serialization: bool=True, variant: Optional[str]=None, push_to_hub:
    bool=False, **kwargs):
    """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~DiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a pipeline to. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
    model_index_dict = dict(self.config)
    model_index_dict.pop('_class_name', None)
    model_index_dict.pop('_diffusers_version', None)
    model_index_dict.pop('_module', None)
    model_index_dict.pop('_name_or_path', None)
    if push_to_hub:
        commit_message = kwargs.pop('commit_message', None)
        private = kwargs.pop('private', False)
        create_pr = kwargs.pop('create_pr', False)
        token = kwargs.pop('token', None)
        repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
        repo_id = create_repo(repo_id, exist_ok=True, private=private,
            token=token).repo_id
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
            sub_model = _unwrap_model(sub_model)
            model_cls = sub_model.__class__
        save_method_name = None
        for library_name, library_classes in LOADABLE_CLASSES.items():
            if library_name in sys.modules:
                library = importlib.import_module(library_name)
            else:
                logger.info(
                    f'{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}'
                    )
            for base_class, save_load_methods in library_classes.items():
                class_candidate = getattr(library, base_class, None)
                if class_candidate is not None and issubclass(model_cls,
                    class_candidate):
                    save_method_name = save_load_methods[0]
                    break
            if save_method_name is not None:
                break
        if save_method_name is None:
            logger.warning(
                f'self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved.'
                )
            self.register_to_config(**{pipeline_component_name: (None, None)})
            continue
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
    self.save_config(save_directory)
    if push_to_hub:
        model_card = load_or_create_model_card(repo_id, token=token,
            is_pipeline=True)
        model_card = populate_model_card(model_card)
        model_card.save(os.path.join(save_directory, 'README.md'))
        self._upload_folder(save_directory, repo_id, token=token,
            commit_message=commit_message, create_pr=create_pr)

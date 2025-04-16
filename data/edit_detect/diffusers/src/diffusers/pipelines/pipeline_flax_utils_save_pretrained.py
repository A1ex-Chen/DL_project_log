def save_pretrained(self, save_directory: Union[str, os.PathLike], params:
    Union[Dict, FrozenDict], push_to_hub: bool=False, **kwargs):
    """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~FlaxDiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
    self.save_config(save_directory)
    model_index_dict = dict(self.config)
    model_index_dict.pop('_class_name')
    model_index_dict.pop('_diffusers_version')
    model_index_dict.pop('_module', None)
    if push_to_hub:
        commit_message = kwargs.pop('commit_message', None)
        private = kwargs.pop('private', False)
        create_pr = kwargs.pop('create_pr', False)
        token = kwargs.pop('token', None)
        repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
        repo_id = create_repo(repo_id, exist_ok=True, private=private,
            token=token).repo_id
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
        if push_to_hub:
            self._upload_folder(save_directory, repo_id, token=token,
                commit_message=commit_message, create_pr=create_pr)

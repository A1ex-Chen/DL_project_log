def check_model_type(self, supported_models: Union[List[str], dict]):
    """
        Check if the model class is in supported by the pipeline.

        Args:
            supported_models (:obj:`List[str]` or :obj:`dict`):
                The list of models supported by the pipeline, or a dictionary with model class values.
        """
    if not isinstance(supported_models, list):
        supported_models = [item[1].__name__ for item in supported_models.
            items()]
    if self.model.__class__.__name__ not in supported_models:
        raise PipelineException(self.task, self.model.base_model_prefix,
            f"The model '{self.model.__class__.__name__}' is not supported for {self.task}. Supported models are {supported_models}"
            )

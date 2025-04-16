def __init__(self, model: Union[str, Path]='yolov8n.pt', task: str=None,
    verbose: bool=False) ->None:
    """
        Initializes a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of model
        sources, including local files, Ultralytics HUB models, and Triton Server models. The method initializes several
        important attributes of the model and prepares it for operations like training, prediction, or export.

        Args:
            model (Union[str, Path], optional): The path or model file to load or create. This can be a local
                file path, a model name from Ultralytics HUB, or a Triton Server model. Defaults to 'yolov8n.pt'.
            task (Any, optional): The task type associated with the YOLO model, specifying its application domain.
                Defaults to None.
            verbose (bool, optional): If True, enables verbose output during the model's initialization and subsequent
                operations. Defaults to False.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.
        """
    super().__init__()
    self.callbacks = callbacks.get_default_callbacks()
    self.predictor = None
    self.model = None
    self.trainer = None
    self.ckpt = None
    self.cfg = None
    self.ckpt_path = None
    self.overrides = {}
    self.metrics = None
    self.session = None
    self.task = task
    model = str(model).strip()
    if self.is_hub_model(model):
        checks.check_requirements('hub-sdk>=0.0.8')
        self.session = HUBTrainingSession.create_session(model)
        model = self.session.model_file
    elif self.is_triton_model(model):
        self.model_name = self.model = model
        return
    if Path(model).suffix in {'.yaml', '.yml'}:
        self._new(model, task=task, verbose=verbose)
    else:
        self._load(model, task=task)

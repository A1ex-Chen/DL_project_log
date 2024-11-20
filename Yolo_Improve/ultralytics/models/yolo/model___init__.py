def __init__(self, model='yolov8s-world.pt', verbose=False) ->None:
    """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str | Path): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
    super().__init__(model=model, task='detect', verbose=verbose)
    if not hasattr(self.model, 'names'):
        self.model.names = yaml_load(ROOT / 'cfg/datasets/coco8.yaml').get(
            'names')

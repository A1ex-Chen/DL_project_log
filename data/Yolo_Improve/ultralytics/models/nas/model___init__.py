def __init__(self, model='yolo_nas_s.pt') ->None:
    """Initializes the NAS model with the provided or default 'yolo_nas_s.pt' model."""
    assert Path(model).suffix not in {'.yaml', '.yml'
        }, 'YOLO-NAS models only support pre-trained models.'
    super().__init__(model, task='detect')

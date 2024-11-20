def __init__(self, model='FastSAM-x.pt'):
    """Call the __init__ method of the parent class (YOLO) with the updated default model."""
    if str(model) == 'FastSAM.pt':
        model = 'FastSAM-x.pt'
    assert Path(model).suffix not in {'.yaml', '.yml'
        }, 'FastSAM models only support pre-trained models.'
    super().__init__(model=model, task='segment')

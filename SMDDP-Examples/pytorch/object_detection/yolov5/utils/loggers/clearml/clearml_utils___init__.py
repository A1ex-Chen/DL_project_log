def __init__(self, opt, hyp):
    """
        - Initialize ClearML Task, this object will capture the experiment
        - Upload dataset version to ClearML Data if opt.upload_dataset is True

        arguments:
        opt (namespace) -- Commandline arguments for this run
        hyp (dict) -- Hyperparameters for this run

        """
    self.current_epoch = 0
    self.current_epoch_logged_images = set()
    self.max_imgs_to_log_per_epoch = 16
    self.bbox_interval = opt.bbox_interval
    self.clearml = clearml
    self.task = None
    self.data_dict = None
    if self.clearml:
        self.task = Task.init(project_name='YOLOv5', task_name='training',
            tags=['YOLOv5'], output_uri=True, auto_connect_frameworks={
            'pytorch': False})
        self.task.connect(hyp, name='Hyperparameters')
        if opt.data.startswith('clearml://'):
            self.data_dict = construct_dataset(opt.data)
            opt.data = self.data_dict

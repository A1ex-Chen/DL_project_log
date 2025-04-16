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
        self.task = Task.init(project_name=opt.project if opt.project !=
            'runs/train' else 'YOLOv5', task_name=opt.name if opt.name !=
            'exp' else 'Training', tags=['YOLOv5'], output_uri=True,
            reuse_last_task_id=opt.exist_ok, auto_connect_frameworks={
            'pytorch': False})
        self.task.connect(hyp, name='Hyperparameters')
        self.task.connect(opt, name='Args')
        self.task.set_base_docker('ultralytics/yolov5:latest',
            docker_arguments=
            '--ipc=host -e="CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1"',
            docker_setup_bash_script='pip install clearml')
        if opt.data.startswith('clearml://'):
            self.data_dict = construct_dataset(opt.data)
            opt.data = self.data_dict

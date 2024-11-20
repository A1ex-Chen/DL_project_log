def __init__(self, opt, hyp, run_id=None, job_type='Training', **
    experiment_kwargs) ->None:
    self.job_type = job_type
    self.opt = opt
    self.hyp = hyp
    self.comet_mode = COMET_MODE
    self.save_model = opt.save_period > -1
    self.model_name = COMET_MODEL_NAME
    self.log_batch_metrics = COMET_LOG_BATCH_METRICS
    self.comet_log_batch_interval = COMET_BATCH_LOGGING_INTERVAL
    self.upload_dataset = self.opt.upload_dataset or COMET_UPLOAD_DATASET
    self.resume = self.opt.resume
    self.default_experiment_kwargs = {'log_code': False, 'log_env_gpu': 
        True, 'log_env_cpu': True, 'project_name': COMET_PROJECT_NAME}
    self.default_experiment_kwargs.update(experiment_kwargs)
    self.experiment = self._get_experiment(self.comet_mode, run_id)
    self.experiment.set_name(self.opt.name)
    self.data_dict = self.check_dataset(self.opt.data)
    self.class_names = self.data_dict['names']
    self.num_classes = self.data_dict['nc']
    self.logged_images_count = 0
    self.max_images = COMET_MAX_IMAGE_UPLOADS
    if run_id is None:
        self.experiment.log_other('Created from', 'YOLOv5')
        if not isinstance(self.experiment, comet_ml.OfflineExperiment):
            workspace, project_name, experiment_id = self.experiment.url.split(
                '/')[-3:]
            self.experiment.log_other('Run Path',
                f'{workspace}/{project_name}/{experiment_id}')
        self.log_parameters(vars(opt))
        self.log_parameters(self.opt.hyp)
        self.log_asset_data(self.opt.hyp, name='hyperparameters.json',
            metadata={'type': 'hyp-config-file'})
        self.log_asset(f'{self.opt.save_dir}/opt.yaml', metadata={'type':
            'opt-config-file'})
    self.comet_log_confusion_matrix = COMET_LOG_CONFUSION_MATRIX
    if hasattr(self.opt, 'conf_thres'):
        self.conf_thres = self.opt.conf_thres
    else:
        self.conf_thres = CONF_THRES
    if hasattr(self.opt, 'iou_thres'):
        self.iou_thres = self.opt.iou_thres
    else:
        self.iou_thres = IOU_THRES
    self.log_parameters({'val_iou_threshold': self.iou_thres,
        'val_conf_threshold': self.conf_thres})
    self.comet_log_predictions = COMET_LOG_PREDICTIONS
    if self.opt.bbox_interval == -1:
        self.comet_log_prediction_interval = (1 if self.opt.epochs < 10 else
            self.opt.epochs // 10)
    else:
        self.comet_log_prediction_interval = self.opt.bbox_interval
    if self.comet_log_predictions:
        self.metadata_dict = {}
        self.logged_image_names = []
    self.comet_log_per_class_metrics = COMET_LOG_PER_CLASS_METRICS
    self.experiment.log_others({'comet_mode': COMET_MODE,
        'comet_max_image_uploads': COMET_MAX_IMAGE_UPLOADS,
        'comet_log_per_class_metrics': COMET_LOG_PER_CLASS_METRICS,
        'comet_log_batch_metrics': COMET_LOG_BATCH_METRICS,
        'comet_log_confusion_matrix': COMET_LOG_CONFUSION_MATRIX,
        'comet_model_name': COMET_MODEL_NAME})
    if hasattr(self.opt, 'comet_optimizer_id'):
        self.experiment.log_other('optimizer_id', self.opt.comet_optimizer_id)
        self.experiment.log_other('optimizer_objective', self.opt.
            comet_optimizer_objective)
        self.experiment.log_other('optimizer_metric', self.opt.
            comet_optimizer_metric)
        self.experiment.log_other('optimizer_parameters', json.dumps(self.hyp))

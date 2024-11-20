def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    logger = WandbLogger(opt, '', None, data, job_type='Dataset Creation')

def check_wandb_resume(opt):
    process_wandb_config_ddp_mode(opt) if opt.global_rank not in [-1, 0
        ] else None
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            if opt.global_rank not in [-1, 0]:
                run_id, project, model_artifact_name = get_run_info(opt.resume)
                api = wandb.Api()
                artifact = api.artifact(project + '/' + model_artifact_name +
                    ':latest')
                modeldir = artifact.download()
                opt.weights = str(Path(modeldir) / 'last.pt')
            return True
    return None

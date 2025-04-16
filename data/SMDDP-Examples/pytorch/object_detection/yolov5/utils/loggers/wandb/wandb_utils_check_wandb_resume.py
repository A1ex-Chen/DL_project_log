def check_wandb_resume(opt):
    process_wandb_config_ddp_mode(opt) if RANK not in [-1, 0] else None
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            if RANK not in [-1, 0]:
                entity, project, run_id, model_artifact_name = get_run_info(opt
                    .resume)
                api = wandb.Api()
                artifact = api.artifact(entity + '/' + project + '/' +
                    model_artifact_name + ':latest')
                modeldir = artifact.download()
                opt.weights = str(Path(modeldir) / 'last.pt')
            return True
    return None

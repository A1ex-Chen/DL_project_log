def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return run_id, project, model_artifact_name

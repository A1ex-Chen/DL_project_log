def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f'using task specific params for {task}: {pars}')
        model.config.update(pars)

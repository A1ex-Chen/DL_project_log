def save_checkpoint(args, accelerator, global_step):
    output_dir = args.output_dir
    if (accelerator.is_main_process and args.checkpoints_total_limit is not
        None):
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))
        if len(checkpoints) >= args.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f'{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints'
                )
            logger.info(
                f"removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir,
                    removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    save_path = Path(output_dir) / f'checkpoint-{global_step}'
    accelerator.save_state(save_path)
    logger.info(f'Saved state to {save_path}')

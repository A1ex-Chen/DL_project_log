def checkpoint(args, global_step, accelerator):
    save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
    accelerator.save_state(save_path)
    logger.info(f'Saved state to {save_path}')

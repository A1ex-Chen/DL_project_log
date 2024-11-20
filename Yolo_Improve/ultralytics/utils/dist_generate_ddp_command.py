def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)
    file = generate_ddp_file(trainer)
    dist_cmd = ('torch.distributed.run' if TORCH_1_9 else
        'torch.distributed.launch')
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node',
        f'{world_size}', '--master_port', f'{port}', file]
    return cmd, file

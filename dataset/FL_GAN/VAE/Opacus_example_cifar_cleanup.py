def cleanup():
    torch.distributed.destroy_process_group()

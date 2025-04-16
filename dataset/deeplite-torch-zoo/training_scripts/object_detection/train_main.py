def main(opt):
    if RANK in {-1, 0}:
        print_args(vars(opt))
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name,
        exist_ok=opt.exist_ok))
    device = select_device(opt.device, batch=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count(
            ) > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else
            'gloo')
    train(opt, device)

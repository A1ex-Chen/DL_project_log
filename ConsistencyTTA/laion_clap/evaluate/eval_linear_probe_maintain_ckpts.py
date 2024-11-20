def maintain_ckpts(args, startidx, all_idx_len):
    for i in reversed(range(startidx, all_idx_len)):
        if os.path.exists(os.path.join(args.checkpoint_path,
            f'epoch_top_{i}.pt')):
            os.rename(os.path.join(args.checkpoint_path,
                f'epoch_top_{i}.pt'), os.path.join(args.checkpoint_path,
                f'epoch_top_{i + 1}.pt'))
    if os.path.exists(os.path.join(args.checkpoint_path,
        f'epoch_top_{all_idx_len}.pt')):
        os.remove(os.path.join(args.checkpoint_path,
            f'epoch_top_{all_idx_len}.pt'))
    return

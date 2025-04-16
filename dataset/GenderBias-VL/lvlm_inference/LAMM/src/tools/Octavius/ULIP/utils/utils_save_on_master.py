def save_on_master(state, is_best, output_dir):
    if is_main_process():
        best_path = f'{output_dir}/checkpoint_best.pt'
        last_path = f'{output_dir}/checkpoint_last.pt'
        if is_best:
            torch.save(state, best_path)
        if state['epoch'] == 'last':
            torch.save(state, last_path)

def get_best_checkpoint(ckpt_dir, key='mIoU_test', key1='pacc_test', key2=
    'macc_test'):
    ckpt_path = None
    log_file = os.path.join(ckpt_dir, 'logs.csv')
    if os.path.exists(log_file):
        data = pd.read_csv(log_file)
        idx = data[key].idxmax()
        idx1 = data[key1].idxmax()
        idx2 = data[key2].idxmax()
        miou = data[key][idx]
        pacc = data[key1[idx1]]
        macc = data[key2][idx2]
        epoch = data.epoch[idx]
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_epoch_{epoch}.pth')
    assert ckpt_path is not None, f'No trainings found at {ckpt_dir}'
    assert os.path.exists(ckpt_path
        ), f'There is no weights file named {ckpt_path}'
    print(f'Best mIoU: {100 * miou:0.2f} at epoch: {epoch}')
    print(f'Best pacc: {100 * pacc:0.2f} at epoch: {epoch}')
    print(f'Best macc: {100 * macc:0.2f} at epoch: {epoch}')
    return ckpt_path

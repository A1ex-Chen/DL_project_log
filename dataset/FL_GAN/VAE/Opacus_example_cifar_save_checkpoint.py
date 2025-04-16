def save_checkpoint(state, is_best, filename='checkpoint.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

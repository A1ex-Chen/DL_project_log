def save_ckpt(ckpt_dir, model, optimizer, epoch, is_best=False):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':
        optimizer.state_dict()}
    if is_best:
        ckpt_model_filename = 'ckpt_epoch_{}.pth'.format(epoch)
        path = os.path.join(ckpt_dir, ckpt_model_filename)
        torch.save(state, path)
        print('{:>2} has been successfully saved'.format(path), flush=True)
    torch.save(state, os.path.join(ckpt_dir, 'ckpt_latest.pth'))

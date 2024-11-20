def save_ckpt(ckpt_dir, model, optimizer, epoch):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':
        optimizer.state_dict()}
    ckpt_model_filename = 'ckpt_epoch_{}.pth'.format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

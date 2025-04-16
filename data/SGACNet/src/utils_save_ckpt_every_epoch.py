def save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
    best_miou_epoch, best_pacc, best_pacc_epoch, best_macc, best_macc_epoch):
    state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer':
        optimizer.state_dict(), 'best_miou': best_miou, 'best_miou_epoch':
        best_miou_epoch, 'best_pacc': best_pacc, 'best_pacc_epoch':
        best_pacc_epoch, 'best_pacc': best_macc, 'best_pacc_epoch':
        best_macc_epoch}
    ckpt_model_filename = 'ckpt_latest.pth'.format(epoch)
    path = os.path.join(ckpt_dir, ckpt_model_filename)
    torch.save(state, path)
    print('{:>2} has been successfully saved'.format(path))

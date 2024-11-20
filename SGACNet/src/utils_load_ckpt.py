def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("=> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage,
                loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(model_file,
            checkpoint['epoch']))
        epoch = checkpoint['epoch']
        if 'best_miou' in checkpoint:
            best_miou = checkpoint['best_miou']
            print('Best mIoU:', best_miou)
        else:
            best_miou = 0
        if 'best_pacc' in checkpoint:
            best_pacc = checkpoint['best_pacc']
            print('Best pacc:', best_pacc)
        else:
            best_pacc = 0
        if 'best_macc' in checkpoint:
            best_macc = checkpoint['best_macc']
            print('Best macc:', best_macc)
        else:
            best_macc = 0
        if 'best_miou_epoch' in checkpoint:
            best_miou_epoch = checkpoint['best_miou_epoch']
            print('Best mIoU epoch:', best_miou_epoch)
        else:
            best_miou_epoch = 0
        if 'best_pacc_epoch' in checkpoint:
            best_pacc_epoch = checkpoint['best_pacc_epoch']
            print('Best pacc epoch:', best_pacc_epoch)
        else:
            best_pacc_epoch = 0
        if 'best_macc_epoch' in checkpoint:
            best_pacc_epoch = checkpoint['best_macc_epoch']
            print('Best macc epoch:', best_macc_epoch)
        else:
            best_macc_epoch = 0
        return (epoch, best_miou, best_miou_epoch, best_pacc,
            best_pacc_epoch, best_macc, best_macc_epoch)
    else:
        print("=> no checkpoint found at '{}'".format(model_file))
        sys.exit(1)

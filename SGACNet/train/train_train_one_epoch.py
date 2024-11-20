def train_one_epoch(model, train_loader, device, optimizer,
    loss_function_train, epoch, lr_scheduler, modality,
    label_downsampling_rates, debug_mode=False):
    training_start_time = time.time()
    lr_scheduler.step(epoch)
    samples_of_epoch = 0
    model.train()
    losses_list = []
    total_loss_list = []
    for i, sample in enumerate(train_loader):
        start_time_for_one_step = time.time()
        if modality in ['rgbd', 'rgb']:
            image = sample['image'].to(device)
            batch_size = image.data.shape[0]
        if modality in ['rgbd', 'depth']:
            depth = sample['depth'].to(device)
            batch_size = depth.data.shape[0]
        target_scales = [sample['label'].to(device)]
        if len(label_downsampling_rates) > 0:
            for rate in sample['label_down']:
                target_scales.append(sample['label_down'][rate].to(device))
        for param in model.parameters():
            param.grad = None
        if modality == 'rgbd':
            pred_scales = model(image, depth)
        elif modality == 'rgb':
            pred_scales = model(image)
        else:
            pred_scales = model(depth)
        losses = loss_function_train(pred_scales, target_scales)
        loss_segmentation = sum(losses)
        total_loss = loss_segmentation
        total_loss.backward()
        optimizer.step()
        losses_list.append([loss.cpu().detach().numpy() for loss in losses])
        total_loss = total_loss.cpu().detach().numpy()
        total_loss_list.append(total_loss)
        if np.isnan(total_loss):
            raise ValueError('Loss is None')
        samples_of_epoch += batch_size
        time_inter = time.time() - start_time_for_one_step
        learning_rates = lr_scheduler.get_lr()
        print_log(epoch, samples_of_epoch, batch_size, len(train_loader.
            dataset), total_loss, time_inter, learning_rates)
        if debug_mode:
            break
    logs = dict()
    logs['time_training'] = time.time() - training_start_time
    logs['loss_train_total'] = np.mean(total_loss_list)
    losses_train = np.mean(losses_list, axis=0)
    logs['loss_train_full_size'] = losses_train[0]
    for i, rate in enumerate(label_downsampling_rates):
        logs['loss_train_down_{}'.format(rate)] = losses_train[i + 1]
    logs['epoch'] = epoch
    for i, lr in enumerate(learning_rates):
        logs['lr_{}'.format(i)] = lr
    return logs

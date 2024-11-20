def get_data_from_log(txt_path):
    """
    Output dictionary from out.txt log file
    """
    with open(txt_path) as f:
        lines = f.readlines()
    val_data = {}
    train_data = {}
    train_losses = []
    train_losses_epoch = []
    for i in range(len(lines)):
        if '| INFO |' in lines[i]:
            if 'Eval Epoch' in lines[i]:
                if 'val_loss' in lines[i]:
                    line = lines[i].split('Eval Epoch: ')[-1]
                    num_epoch = int(line.split('\t')[0].split(' ')[0])
                    d = {line.split('\t')[0].split(' ')[1].replace(':', ''):
                        float(line.split('\t')[0].split(' ')[-1])}
                    for i in range(1, len(line.split('\t'))):
                        d = save_to_dict(line.split('\t')[i], d)
                    val_data[num_epoch] = d
            elif 'Train Epoch' in lines[i]:
                num_epoch = int(lines[i].split('Train Epoch: ')[1][0])
                loss = float(lines[i].split('Loss: ')[-1].split(' (')[0])
                train_losses.append(loss)
                train_losses_epoch.append(num_epoch)
    for i in range(len(train_losses)):
        train_data[i] = {'num_epoch': train_losses_epoch[i], 'train_loss':
            train_losses[i]}
    return train_data, val_data

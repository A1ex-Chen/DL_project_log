def print_log(epoch, local_count, count_inter, dataset_size, loss,
    time_inter, learning_rates):
    print_string = 'Train Epoch: {:>3} [{:>4}/{:>4} ({: 5.1f}%)]'.format(epoch,
        local_count, dataset_size, 100.0 * local_count / dataset_size)
    for i, lr in enumerate(learning_rates):
        print_string += '   lr_{}: {:>6}'.format(i, round(lr, 10))
    print_string += '   Loss: {:0.6f}'.format(loss.item())
    print_string += '  [{:0.2f}s every {:>4} data]'.format(time_inter,
        count_inter)
    print(print_string, flush=True)

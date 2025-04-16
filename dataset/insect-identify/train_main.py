def main():
    input_arguments = process_arguments()
    default_device = torch.device('cuda' if torch.cuda.is_available() and
        input_arguments.gpu else 'cpu')
    torch.cuda.empty_cache()
    input_size = 9216
    choosen_architecture = input_arguments.choosen_archi
    if choosen_architecture[:3] == 'vgg':
        input_size = 25088
    if choosen_architecture[:8] == 'densenet':
        input_size = 1024
    (train_data, test_data, valid_data, trainloader, testloader, validloader
        ) = du.loading_data(input_arguments.data_directory)
    model = mu.set_pretrained_model(choosen_architecture)
    model = mu.set_model_classifier(model, input_arguments.hidden_units,
        input_size, output_size=15, dropout=0.1)
    model, epochs, optimizer = mu.train_model(model, trainloader,
        input_arguments.epochs, validloader, input_arguments.learning_rate,
        default_device, choosen_architecture)
    if not os.path.exists(input_arguments.save_directory):
        os.makedirs(input_arguments.save_directory)
    checkpoint_file_path = os.path.join(input_arguments.save_directory, 
        choosen_architecture + '_102improve' + str(input_arguments.epochs) +
        '.pth')
    mu.create_checkpoint(model, input_arguments.choosen_archi, train_data,
        epochs, optimizer, checkpoint_file_path, input_size, output_size=15)
    pass

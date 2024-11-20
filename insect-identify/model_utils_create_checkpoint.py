def create_checkpoint(model, model_name, train_data, epochs, optimizer,
    checkpoint_file_path, input_size=25088, output_size=102):
    """ Create a checkpoint file for the given model
    """
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'input_size': input_size, 'output_size': output_size,
        'epochs': epochs, 'model_name': model_name, 'classifier': model.
        classifier, 'class_to_idx': model.class_to_idx, 'optimizer_state':
        optimizer.state_dict(), 'state_dict': model.state_dict()}
    torch.save(checkpoint, checkpoint_file_path)
    return

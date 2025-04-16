def main():
    input_arguments = process_arguments()
    default_device = torch.device('cuda' if torch.cuda.is_available() and
        input_arguments.gpu else 'cpu')
    probs, classes = mu.predict(input_arguments.input_image_path,
        input_arguments.checkpoint_file_path, default_device,
        input_arguments.topk)
    i = 0
    for specie in classes:
        print('your dataset named : ' + specie +
            ' predicted with probability: ' + str(probs[i]))
        i += 1
    pass

def predict_by_file(input_image_path, checkpoint_file_path=
    'checkpoint_dir\\vgg16_CUDA5.pth', default_device='gpu', topk=2):
    default_device = torch.device('cuda' if torch.cuda.is_available() else
        'cpu')
    probs, classes = mu.predict(input_image_path, checkpoint_file_path,
        default_device, topk)
    i = 0
    for specie in classes:
        print('your dataset named : ' + specie +
            ' predicted with probability: ' + str(probs[i]))
        i += 1

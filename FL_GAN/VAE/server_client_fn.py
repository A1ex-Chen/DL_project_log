def client_fn(cid: str) ->Fl_Client:
    my_device = ''
    try:
        if torch.backends.mps.is_built():
            my_device = 'mps'
    except AttributeError:
        if torch.cuda.is_available():
            my_device = 'cuda:0'
        else:
            my_device = 'cpu'
    DEVICE = torch.device(my_device)
    global DATASET, CLIENT
    model = VAE(DATASET)
    trainset, testset = load_partition(CLIENT, DATASET)
    client_trainloader = DataLoader(trainset, PARAMS['batch_size'])
    client_testloader = DataLoader(test, PARAMS['batch_size'])
    sample_rate = PARAMS['batch_size'] / len(trainset)
    return Fl_Client(cid, client_trainloader, client_testloader, device=DEVICE)

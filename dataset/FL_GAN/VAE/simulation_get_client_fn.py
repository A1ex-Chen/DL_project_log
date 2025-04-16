def get_client_fn(args):

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
        DEVICE = my_device
        DATASET = args.dataset
        CLIENT = args.client
        batch_size = args.batch_size
        model = VAE(DATASET)
        trainset, testset = load_partition(CLIENT, DATASET)
        client_trainloader = DataLoader(trainset, batch_size)
        client_testloader = DataLoader(testset, batch_size)
        sample_rate = args.sample_rate
        return Fl_Client(cid, model, client_trainloader, client_testloader,
            sample_rate, device=DEVICE)
    return client_fn

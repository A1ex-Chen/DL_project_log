def print_size_of_model(model):
    torch.save(model.state_dict(), 'temp.p')
    print('Size (MB):', os.path.getsize('temp.p') / 1000000.0)
    os.remove('temp.p')
